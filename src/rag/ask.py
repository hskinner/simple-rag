import ollama

from consts import INSTRUCTION_PREFIX

def model_chat(model: str, query: str, context_chunks: list[dict[str, any]]):
    instruction_prompt = (
        INSTRUCTION_PREFIX
        + '\n'
        + '\n'.join(f" - {row['content']}" for row in context_chunks)
    )

    stream = ollama.chat(
        model=model,
        messages=[
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': query},
        ],
        stream=True,
    )

    print(f'{model} response:')
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)
    
    print('\n')
