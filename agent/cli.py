import click
from agent import run_agent


@click.command()
@click.argument('query')
@click.option('--backend', 'backend_name', default='ollama',
              type=click.Choice(['anthropic', 'openai', 'ollama']), show_default=True)
@click.option('--model', default=None, help='Model name (uses backend default if omitted)')
@click.option('--memory-strategy', 'memory_strategy', default='full',
              type=click.Choice(['full', 'sliding', 'summarize']), show_default=True,
              help='How to manage conversation context as it grows')
def main(query: str, backend_name: str, model: str | None, memory_strategy: str) -> None:
    if backend_name == 'anthropic':
        from backends.anthropic import AnthropicBackend
        backend = AnthropicBackend(model or 'claude-sonnet-4-5')
    elif backend_name == 'openai':
        from backends.openai import OpenAIBackend
        backend = OpenAIBackend(model or 'gpt-4o')
    else:  # ollama
        from backends.openai import OpenAIBackend
        backend = OpenAIBackend(model or 'llama3.1:8b', base_url='http://localhost:11434/v1')

    run_agent(query, backend, memory_strategy=memory_strategy)


if __name__ == '__main__':
    main()
