import numpy as np

from pathlib import Path

import ollama
import psycopg

from context import Context
from consts import EMBEDDING_MODEL
from db import write_db


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

    def insert_rows(cursor: psycopg.Cursor):
        cursor.executemany(
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
    
    write_db(context, insert_rows)
    
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
