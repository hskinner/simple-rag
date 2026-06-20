from pathlib import Path

import click
import subprocess
import time

import psycopg
from pgvector.psycopg import register_vector

from ask import model_chat
from consts import CONTAINER_NAME
from consts import LANGUAGE_MODEL
from consts import DEFAULT_TABLE
from consts import POSTGRES_IMAGE
from consts import EMBEDDING_DIMENSIONS
from context import Context, load_env
from ingestion import load_file
from search import similarity_search


@click.group()
def cli():
    ...


@cli.command()
@click.option('--table', default=DEFAULT_TABLE)
@click.option('--embedding-dimensions', default=EMBEDDING_DIMENSIONS, type=int)
def setup(table: str, embedding_dimensions: int):
    context = load_env()

    existing = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={CONTAINER_NAME}", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    if existing == CONTAINER_NAME:
        print(f"Container {CONTAINER_NAME!r} already exists. Starting it...")
        run(["docker", "start", CONTAINER_NAME])
    else:
        print(f"Creating Postgres container {CONTAINER_NAME!r}...")
        run([
            "docker", "run",
            "--name", CONTAINER_NAME,
            "-e", f"POSTGRES_USER={context.postgres_user}",
            "-e", f"POSTGRES_PASSWORD={context.postgres_password}",
            "-e", f"POSTGRES_DB={context.postgres_db}",
            "-p", f"{context.postgres_port}:5432",
            "-d",
            POSTGRES_IMAGE,
        ])

    wait_for_postgres(context)
    init_schema(context, table, embedding_dimensions)


@cli.command()
@click.option('--table', default=DEFAULT_TABLE)
@click.option('--filepath', required=True)
@click.option('--embedding-dimensions', default=EMBEDDING_DIMENSIONS, type=int)
def load(table: str, filepath: str, embedding_dimensions: int):
    context = load_env()

    wait_for_postgres(context)
    init_schema(context, table, embedding_dimensions)

    load_file(Path(filepath), context, table)


@cli.command()
@click.option('--table', default=DEFAULT_TABLE)
@click.option('--model', default=LANGUAGE_MODEL)
@click.option('--top-k', 'top_k', type=int, default=5)
@click.option('--query')
def query(table: str, model: str, top_k: int, query: str | None):
    context = load_env()

    wait_for_postgres(context)

    if query:
        retrieved_knowledge = similarity_search(context, table, query, top_k)
        model_chat(model, query, retrieved_knowledge)
        return
    
    while True:
        input_query = input('Ask me a question: ')
        if input_query.strip().lower() == 'exit':
            print('Exiting...')
            return

        retrieved_knowledge = similarity_search(context, table, input_query, top_k)
        model_chat(model, query, retrieved_knowledge)


@cli.command()
@click.option('--table', default=DEFAULT_TABLE)
def clean(table: str):
    context = load_env()

    wait_for_postgres(context)

    with psycopg.connect(context.database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP TABLE IF EXISTS {table}')
    
    print(f'Dropped {table}')


def run(cmd: list[str]):
    print('+', ' '.join(cmd))
    subprocess.run(cmd, check=True)


def wait_for_postgres(context: Context, timeout_seconds: int = 45):
    print("Waiting for Postgres to accept connections...")
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        try:
            with psycopg.connect(context.database_url()) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
            print("Postgres is ready.")
            return
        except Exception:
            time.sleep(1)

    raise TimeoutError("Postgres did not become ready in time.")


def init_schema(context: Context, table: str, embedding_dimensions: int):
    with psycopg.connect(context.database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
            register_vector(conn)

            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    id BIGSERIAL PRIMARY KEY,
                    source TEXT,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding VECTOR({embedding_dimensions}) NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT now()
                );
            """)

            # HNSW index for faster approximate nearest-neighbor search.
            # For small datasets, exact search without this index is also fine.
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS text_chunks_embedding_hnsw_idx
                ON {table}
                USING hnsw (embedding vector_cosine_ops);
            """)

        conn.commit()

    print("Schema initialized.")


if __name__ == '__main__':
    cli()
