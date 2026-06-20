import numpy as np

import psycopg
from pgvector.psycopg import register_vector

from context import Context
from ingestion import embed_text

def similarity_search(context: Context, table: str, query: str, top_k: int = 5) -> list[str, float]:
    query_embedding = embed_text([query])[0]

    with psycopg.connect(context.database_url()) as conn:
        register_vector(conn)

        with conn.cursor() as cur:
            embedding_vector = np.array(query_embedding, dtype=np.float32)

            cur.execute(
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

            rows = cur.fetchall()
    
    if not rows:
        print('No results found')
        return []

    return [
        {
            'content': content,
            'cosine_similarity': cosine_similarity,
        }
        for content, cosine_similarity in rows
    ]
