import numpy as np

from pathlib import Path

import ollama
import psycopg
from pgvector.psycopg import register_vector

from context import Context
from consts import EMBEDDING_MODEL


def load_file(path: Path, context: Context, destination_table: str):
    text = path.read_text(encoding='utf-8')
    chunks = chunk_text(text)

    if not chunks:
        print('No text chunks found')
        return

    print(f'Embedding {len(chunks)} chunks...')
    embeddings = embed_text(chunks)

    rows = [
        (
            str(path),
            i,
            chunk,
            np.array(embedding, dtype=np.float32),
        )
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    with psycopg.connect(context.database_url()) as conn:
        register_vector(conn)

        with conn.cursor() as cur:
            cur.executemany(
                f"""
                    INSERT INTO {destination_table}(
                        source,
                        chunk_index,
                        content,
                        embedding
                    )
                    VALUES (%s, %s, %s, %s)
                """,
                rows,
            )

        conn.commit()
    
    print(f'Loaded {len(rows)} chunks from {path}')


def chunk_text(text: str, max_words: int = 220, overlap_words: int = 40) -> list[str]:
    words = text.split()
    if not words:
        return []
    
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(' '.join(words[i:i+max_words]))
        i += max_words - overlap_words
    
    return chunks


def embed_text(texts: list[str]) -> list[list[float]]:
    return [
        ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
        for chunk in texts
    ]
