import asyncio
import ast
import io
from dataclasses import dataclass
import json
import os
import tokenize

import numpy as np
from openai import AsyncOpenAI
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
        """Initialize the async embedding client and tokenizer."""
        self.client = AsyncOpenAI(api_key=API_KEY)
        self.tokenizer = tiktoken.get_encoding(MODEL_NAME)

    def preprocess_chunks(
        self, code_chunks: list[str]
    ) -> tuple[list[str], list[str]]:
        """Split chunks and warn if they exceed the token limit."""

        nl_texts: list[str] = []
        code_texts: list[str] = []
        for chunk in code_chunks:
            nl_chunk, code_chunk = split_code_chunk(chunk)
            nl_tokens = len(self.tokenizer.encode(nl_chunk))
            code_tokens = len(self.tokenizer.encode(code_chunk))
            if nl_tokens > MAX_TOKENS:
                print(
                    f"Warning: natural language chunk has {nl_tokens} tokens which exceeds the maximum of {MAX_TOKENS}.",
                )
            if code_tokens > MAX_TOKENS:
                print(
                    f"Warning: code chunk has {code_tokens} tokens which exceeds the maximum of {MAX_TOKENS}.",
                )
            nl_texts.append(nl_chunk)
            code_texts.append(code_chunk)
        return nl_texts, code_texts

    async def embed(self, text: str) -> np.ndarray:
        """Embed a single text."""

        token_count = len(self.tokenizer.encode(text))
        if token_count > MAX_TOKENS:
            print(
                f"Warning: text has {token_count} tokens which exceeds the maximum of {MAX_TOKENS}.",
            )

        response = await self.client.embeddings.create(
            input=[text],
            model=MODEL_NAME,
            dimensions=EMBEDDING_DIMENSIONS,
            timeout=60,
        )
        return np.array(response.data[0].embedding)

    async def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed many texts in one API call."""

        response = await self.client.embeddings.create(
            input=texts,
            model=MODEL_NAME,
            dimensions=EMBEDDING_DIMENSIONS,
            timeout=60,
        )
        return np.array([d.embedding for d in response.data])

    async def embed_code_pairs(
        self, natural_language_texts: list[str], code_texts: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Embed natural language and code chunks separately."""

        nl_task = self.embed_batch(natural_language_texts)
        code_task = self.embed_batch(code_texts)
        nl_embeddings, code_embeddings = await asyncio.gather(nl_task, code_task)
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
