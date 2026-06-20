import click
import os

import ollama

from dotenv import load_dotenv


REFERENCE_TEXT = 'resources/cat-facts.txt'
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

INSTRUCTION_PREFIX = """You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:"""

@click.command()
@click.option('--model', help='Language model to use', default=LANGUAGE_MODEL)
@click.option('--in-memory-db', 'in_memory_db', is_flag=True, help='Use in memory embeddings DB')
def main(model: str, in_memory_db: bool):
    if in_memory_db:
        db = load_in_memory_db(REFERENCE_TEXT)
    else:
        db = load_postgres_db(REFERENCE_TEXT)

    while True:
        input_query = input('Ask me a question: ')
        if input_query.strip().lower() == 'exit':
            print('Exiting...')
            break

        retrieved_knowledge = retrieve(input_query, db)

        print('Retrieved knowledge:')
        for chunk, similarity in retrieved_knowledge:
            print(f' - (similarity: {similarity:.2f}) {chunk}')

        instruction_prompt = f"""{INSTRUCTION_PREFIX}\n{'\n'.join([f' - {chunk}' for chunk, _ in retrieved_knowledge])}"""
        
        stream = ollama.chat(
            model=model,
            messages=[
                {'role': 'system', 'content': instruction_prompt},
                {'role': 'user', 'content': input_query},
            ],
            stream=True,
        )

        # print the response from the chatbot in real-time
        print('Chatbot response:')
        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)
        
        print('\n')


def load_in_memory_db(filename: str) -> list[tuple[str, list[float]]]:
    in_memory_db = []
    dataset = []

    with open(filename, 'r') as file:
        dataset = file.readlines()
        print(f'Loaded {len(dataset)} entries')

    for line in dataset:
        embedding = ollama.embed(model=EMBEDDING_MODEL, input=line)['embeddings'][0]
        in_memory_db.append((line, embedding))
    
    return in_memory_db


def load_postgres_db(filename: str):
    load_dotenv()

    postgres_user = os.getenv("POSTGRES_USER", "postgres")
    postgres_password = os.environ["POSTGRES_PASSWORD"]
    postgres_db = os.getenv("POSTGRES_DB", "vectordb")
    postgres_port = os.getenv("POSTGRES_PORT", "5432")

    database_url = (
        f'postgresql://{postgres_user}:{postgres_password}'
        f'@localhost:{postgres_port}/{postgres_db}',
    )


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5

    return dot_product / (norm_a * norm_b)


def retrieve(query: str, in_memory_db: list[tuple[str, str]], top_n: int=3) -> list[tuple[str, float]]:
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]

    # temporary list to store (chunk, similarity) pairs
    similarities = []
    for chunk, embedding in in_memory_db:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))
    
    # sort by similarity in descending order, because higher similarity means more relevant chunks
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # finally, return the top N most relevant chunks
    return similarities[:top_n]


if __name__ == '__main__':
    main()
