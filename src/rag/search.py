import numpy as np

import psycopg

from context import Context
from ingestion import embed_text
from db import read_db


def similarity_search(context: Context, table: str, query: str, top_k: int = 5) -> list[dict[str, any]]:
    query_embedding = embed_text([query])[0]

    def impl(cursor: psycopg.Cursor):
        embedding_vector = np.array(query_embedding, dtype=np.float32)

        cursor.execute(
            f"""
                SELECT
                    content,
                    1 - (embedding <=> %s) AS cosine_similarity
                FROM {table}
                ORDER BY 2 DESC
                LIMIT %s
            """,
            (
                embedding_vector,
                top_k,
            ),
        )

    rows = read_db(context, impl)

    return [
        {'content': content, 'cosine_similarity': cosine_similarity}
        for content, cosine_similarity in rows
    ]
