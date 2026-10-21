import json
import time
import click
import agent

from anthropic import Anthropic
from backends.anthropic import AnthropicBackend
from backends.openai import OpenAIBackend
from tools import TOOL_SCHEMAS, REGISTRY

_judge = Anthropic()

with open('evals.json', 'r') as file:
    data = json.load(file)


def llm_judge(question: str, response_text: str, rubric: str) -> tuple[bool, str]:
    messages = [{
        "role": "user",
        "content": (
            f"You are a factual judge evaluating an AI research agent's response. Today's date is June 17, 2026. "
            f"Use the web_search tool to independently verify any time-sensitive facts before making your judgment.\n\n"
            f"Question: {question}\n"
            f"Evaluation rubric: {rubric}\n"
            f"Response:\n{response_text}\n\n"
            "Search the web to verify the key facts, then briefly explain whether the response satisfies each criterion in the rubric. "
            "On the final line, write only YES or NO."
        )
    }]

    while True:
        result = _judge.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        if result.stop_reason == "end_turn":
            text = next(b.text for b in result.content if hasattr(b, "text")).strip()
            verdict = text.split('\n')[-1].strip().upper()
            return verdict.startswith("YES"), text

        messages.append({"role": "assistant", "content": result.content})

        tool_results = []
        for block in result.content:
            if block.type == "tool_use":
                output = REGISTRY[block.name](**block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })

        messages.append({"role": "user", "content": tool_results})

def test_cases(backend) -> str:
    score = 0
    inf_times = []
    num_tool_calls = []
    total = len(data['cases'])

    with open('eval_reasoning.txt', 'w') as log:
        for case in data['cases']:
            start = time.time()
            response = agent.run_once(case['question'], backend)
            elapsed = time.time() - start
            inf_times.append(elapsed)

            passed, reasoning = llm_judge(case['question'], response.response_text, case['rubric'])
            num_tool_calls.append(response.tool_call_count)
            if passed:
                score += 1
            status = "PASS" if passed else "FAIL"
            print(f"[{status}] #{case['id']} (tools: {response.tool_call_count}/{case['min_tool_calls']}) {case['question'][:60]}")

            log.write(f"***** {case['id']} *****\n")
            log.write(f"Question: {case['question']}\n")
            log.write(f"Verdict: {status}\n\n")
            log.write(f"{reasoning}\n\n")
            log.flush()

    return f"\nScore: {score}/{total}.\nAverage time per case: {sum(inf_times)/len(inf_times)}\nTool call count: {num_tool_calls[4:]}\n\n"


@click.command()
@click.option('--backend', 'backend_name', default='anthropic',
              type=click.Choice(['anthropic', 'openai', 'ollama']), show_default=True)
@click.option('--model', default=None, help='Model name (uses backend default if omitted)')
def main(backend_name: str, model: str | None) -> None:
    if backend_name == 'anthropic':
        backend = AnthropicBackend(model or 'claude-sonnet-4-5')
    elif backend_name == 'openai':
        backend = OpenAIBackend(model or 'gpt-4o')
    else:  # ollama
        backend = OpenAIBackend(model or 'llama3.2', base_url='http://localhost:11434/v1')

    print(test_cases(backend))


if __name__ == '__main__':
    main()
