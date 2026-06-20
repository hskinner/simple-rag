import dotenv
import os

from dataclasses import dataclass

@dataclass
class Context:
    postgres_user: str
    postgres_password: str
    postgres_db: str
    postgres_port: int

    def database_url(self) -> str:
        return (
            f'postgresql://{self.postgres_user}:{self.postgres_password}'
            f'@localhost:{self.postgres_port}/{self.postgres_db}'
        )


def load_env() -> Context:
    dotenv.load_dotenv()

    postgres_user = os.getenv("POSTGRES_USER", "postgres")
    postgres_password = os.environ["POSTGRES_PASSWORD"]
    postgres_db = os.getenv("POSTGRES_DB", "vectordb")
    postgres_port = os.getenv("POSTGRES_PORT", "5432")

    return Context(
        postgres_user,
        postgres_password,
        postgres_db,
        postgres_port,
    )
