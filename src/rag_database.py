import asyncio
import ast
import io
from dataclasses import dataclass
import json
import os
import tokenize

import numpy as np
from openai import OpenAI
import polars as pl
import tiktoken

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "text-embedding-3-large"
MAX_TOKENS = 8192  # Max tokens for the embedding model
EMBEDDING_DIMENSIONS = 3072


class DataframeKeys:
    """Keys for the RAG DataFrame"""

    KEY_TITLE = "title"
    KEY_TXT = "text"
    KEY_NL_EMBEDDINGS = "nl_embeddings"
    KEY_CODE_EMBEDDINGS = "code_embeddings"
    KEY_SIMILARITIES = "similarities"


# Define RAG database schema
RAG_SCHEMA = pl.DataFrame(
    schema={
        DataframeKeys.KEY_TITLE: pl.Utf8,
        DataframeKeys.KEY_TXT: pl.Utf8,
        DataframeKeys.KEY_NL_EMBEDDINGS: pl.List(pl.Float64),
        DataframeKeys.KEY_CODE_EMBEDDINGS: pl.List(pl.Float64),
    }
)


def split_code_chunk(code_chunk: str) -> tuple[str, str]:
    """Split a code chunk into natural language and code components.

    The *natural language* part consists of the signature together with any
    docstring and inline comments.  The *code* part contains the signature and
    the executable code with docstrings and comments removed.
    """

    # Parse the chunk to obtain the first definition (function/class)
    try:
        module = ast.parse(code_chunk)
        node = module.body[0]
    except Exception:
        return code_chunk, code_chunk

    lines = code_chunk.splitlines()
    signature = lines[0].strip() if lines else ""

    # Extract docstring
    docstring = ast.get_docstring(node) or ""

    # Collect comments using tokenize
    comments: list[str] = []
    for tok in tokenize.generate_tokens(io.StringIO(code_chunk).readline):
        if tok.type == tokenize.COMMENT:
            comments.append(tok.string)

    natural_language_parts = [signature]
    if docstring:
        natural_language_parts.append(f'"""{docstring}"""')
    natural_language_parts.extend(comments)
    natural_language_chunk = "\n".join(natural_language_parts).strip()

    # Remove docstring from AST and unparse to code-only representation
    try:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(getattr(body[0], "value", None), ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body = body[1:]
                node.body = body
            code_chunk_clean = ast.unparse(node)
        else:
            code_chunk_clean = code_chunk
    except Exception:
        code_chunk_clean = code_chunk

    return natural_language_chunk, code_chunk_clean


@dataclass
class RAGQuery:
    """RAG Query Structure"""

    query: str
    k_queries: int


@dataclass
class RAGResponse:
    """Response from a RAG Database Query"""

    titles: list[str]
    texts: list[str]
    similarities: list[float]

    def to_json(self) -> str:
        """Convert RAGResponse to a JSON string."""
        return json.dumps(
            {
                DataframeKeys.KEY_TITLE: self.titles,
                DataframeKeys.KEY_TXT: self.texts,
                DataframeKeys.KEY_SIMILARITIES: self.similarities,
            },
            ensure_ascii=False,
        )

    def to_polars(self) -> pl.DataFrame:
        """Convert RAGResponse to a Polars DataFrame."""
        return pl.DataFrame(
            {
                DataframeKeys.KEY_SIMILARITIES: self.similarities,
                DataframeKeys.KEY_TITLE: self.titles,
                DataframeKeys.KEY_TXT: self.texts,
            }
        )


class EmbeddingModel:
    """Wrapper for the OpenAI Embedding Model"""

    def __init__(self) -> None:
        """Initialize the embedding model client and tokenizer."""
        self.client = OpenAI(
            api_key=API_KEY,
        )
        self.tokenizer = tiktoken.get_encoding(MODEL_NAME)

    def split_text(self, text_to_split: str, max_tokens: int) -> list[str]:
        """Split text into chunks of a maximum token size."""
        tokens = self.tokenizer.encode(text_to_split)
        token_chunks = [tokens[i : i + max_tokens] for i in range(0, len(tokens), max_tokens)]
        return [self.tokenizer.decode(chunk) for chunk in token_chunks]

    async def embed(self, text: str) -> np.ndarray:
        """
        Asynchronously embed a single text into a dense vector.
        """
        original_tokens = self.tokenizer.encode(text)

        chunks = self.split_text(text, MAX_TOKENS) if len(original_tokens) > MAX_TOKENS else [text]

        # Request embeddings for each chunk asynchronously
        async def get_embedding(chunk):  # noqa
            return await self.client.embeddings.create(
                input=chunk,
                model=MODEL_NAME,
                dimensions=EMBEDDING_DIMENSIONS,
                timeout=60,
            )

        chunk_embeddings = await asyncio.gather(*[get_embedding(chunk) for chunk in chunks])

        if len(chunk_embeddings) == 1:
            final_embedding = np.array(chunk_embeddings[0].data[0].embedding)
        else:
            # Weighted average based on token counts
            chunk_token_counts = [len(self.tokenizer.encode(chunk)) for chunk in chunks]
            total_tokens = sum(chunk_token_counts)
            weights = [count / total_tokens for count in chunk_token_counts]

            final_embedding = np.average(
                [response.data[0].embedding for response in chunk_embeddings],
                axis=0,
                weights=weights,
            )

        return final_embedding

    async def batch_embed(self, texts: list[str]) -> np.ndarray:
        """
        Asynchronously embed a list of texts into dense vectors.

        Returns:
            A 2D numpy array of shape (len(texts), embedding dimension)
        """
        tasks = [self.embed(text) for text in texts]
        all_embeddings = await asyncio.gather(*tasks)
        return np.array(all_embeddings)

    async def embed_code_pairs(
        self, natural_language_texts: list[str], code_texts: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Embed natural language and code chunks separately."""

        nl_embeddings = await self.batch_embed(natural_language_texts)
        code_embeddings = await self.batch_embed(code_texts)
        return nl_embeddings, code_embeddings


class VectorDB:
    """A simple in-memory vector database using Polars and NumPy."""

    def __init__(self, dataframe: pl.DataFrame) -> None:
        if dataframe.schema == RAG_SCHEMA.schema:
            self.database = dataframe
        else:
            error_msg = f"Dataframe does not match the required schema: {RAG_SCHEMA.schema}"
            raise ValueError(error_msg)

    def similarity_search(self, query_vector: np.ndarray, k_queries: int = 20) -> RAGResponse:
        """Search for similar vectors using Reciprocal Rank Fusion (RRF).

        Both natural language and code embeddings are compared against the
        provided query vector. The individual rankings are fused using RRF with
        ``k=60``.
        """

        nl_vectors = np.stack(self.database[DataframeKeys.KEY_NL_EMBEDDINGS].to_list())
        code_vectors = np.stack(self.database[DataframeKeys.KEY_CODE_EMBEDDINGS].to_list())

        # Compute cosine similarities
        query_norm = np.linalg.norm(query_vector)

        def _cos_sim(vectors: np.ndarray) -> np.ndarray:
            dot = np.dot(vectors, query_vector)
            norms = np.linalg.norm(vectors, axis=1)
            return dot / (norms * query_norm)

        nl_sim = _cos_sim(nl_vectors)
        code_sim = _cos_sim(code_vectors)

        # Obtain rankings (higher similarity -> better rank)
        nl_rank = np.argsort(nl_sim)[::-1]
        code_rank = np.argsort(code_sim)[::-1]
        nl_rank_map = {idx: rank for rank, idx in enumerate(nl_rank)}
        code_rank_map = {idx: rank for rank, idx in enumerate(code_rank)}

        k = 60
        rrf_scores = [
            1 / (k + nl_rank_map[i] + 1) + 1 / (k + code_rank_map[i] + 1)
            for i in range(len(self.database))
        ]

        top_indices = np.argsort(rrf_scores)[-k_queries:][::-1]
        df_top_k = self.database[top_indices]

        return RAGResponse(
            titles=df_top_k[DataframeKeys.KEY_TITLE].to_list(),
            texts=df_top_k[DataframeKeys.KEY_TXT].to_list(),
            similarities=[rrf_scores[i] for i in top_indices],
        )


class RagDatabase:
    """Database for Retrieval Augmented Generation (RAG)"""

    def __init__(self, vector_db: VectorDB) -> None:
        self.embedding_model = EmbeddingModel()
        self.vector_db = vector_db

    async def rag_process_query(self, rag_query: RAGQuery) -> RAGResponse:
        """Process RAG query and return relevant results"""
        query_embedding = await self.embedding_model.embed(rag_query.query)
        return self.vector_db.similarity_search(query_embedding, rag_query.k_queries)
