from typing import Callable

import time

import psycopg
from pgvector.psycopg import register_vector

from context import Context

def create(context: Context, callable: Callable):
    with psycopg.connect(context.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            register_vector(conn)

            callable(cur)
        
        conn.commit()


def write_db(context: Context, callable: Callable):
    with psycopg.connect(context.database_url) as conn:
        register_vector(conn)

        with conn.cursor() as cur:
            callable(cur)

        conn.commit()


def read_db(context: Context, callable: Callable):
    rows = []

    with psycopg.connect(context.database_url) as conn:
        register_vector(conn)

        with conn.cursor() as cur:
            callable(cur)
            rows = cur.fetchall()
    
    return rows


def wait_for_postgres(context: Context, timeout_seconds: int = 45):
    print("Waiting for Postgres to accept connections...")
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        try:
            with psycopg.connect(context.database_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
            print("Postgres is ready.")
            return
        except Exception:
            time.sleep(1)

    raise TimeoutError("Postgres did not become ready in time.")

