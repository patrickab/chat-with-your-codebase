import asyncio
from dataclasses import dataclass
import json
import os

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
    KEY_EMBEDDINGS = "embeddings"
    KEY_SIMILARITIES = "similarities"


# Define RAG database schema
RAG_SCHEMA = pl.DataFrame(
    schema={
        DataframeKeys.KEY_TITLE: pl.Utf8,
        DataframeKeys.KEY_TXT: pl.Utf8,
        DataframeKeys.KEY_EMBEDDINGS: pl.List(pl.Float64),
    }
)


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


class VectorDB:
    """A simple in-memory vector database using Polars and NumPy."""

    def __init__(self, dataframe: pl.DataFrame) -> None:
        if dataframe.schema == RAG_SCHEMA.schema:
            self.database = dataframe
        else:
            error_msg = f"Dataframe does not match the required schema: {RAG_SCHEMA.schema}"
            raise ValueError(error_msg)

    def similarity_search(self, query_vector: np.ndarray, k_queries: int = 20) -> RAGResponse:
        """
        Search for similar vectors in the database using cosine similarity.

        Parameters:
            query_vector (np.ndarray): Query vector of shape (embedding_dim,)
            k_queries (int): Number of top similarity results to return.

        Returns:
            RAGResponse
        """
        db_vectors = np.stack(self.database[DataframeKeys.KEY_EMBEDDINGS].to_list())

        # Compute cosine similarity
        dot_product = np.dot(db_vectors, query_vector)
        db_norms = np.linalg.norm(db_vectors, axis=1)
        query_norm = np.linalg.norm(query_vector)
        cosine_similarities = dot_product / (db_norms * query_norm)

        # Get top-k indices
        top_indices = np.argsort(cosine_similarities)[-k_queries:][::-1]
        df_top_k = self.database[top_indices]

        return RAGResponse(
            titles=df_top_k[DataframeKeys.KEY_TITLE].to_list(),
            texts=df_top_k[DataframeKeys.KEY_TXT].to_list(),
            similarities=cosine_similarities[top_indices].tolist(),
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
