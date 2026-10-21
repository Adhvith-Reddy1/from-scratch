import time
from dataclasses import dataclass
from typing import Any, Callable

from tools import REGISTRY
from backends.base import Backend, StreamResult
from grounding import verify_grounding, print_grounding_report
from memory import apply_memory, print_memory_stats
from persistent_memory import on_conversation_start, on_conversation_end


@dataclass
class RunResult:
    response_text: str
    tool_call_count: int


def call_with_backoff(fn: Callable[[], Any], max_retries: int = 3) -> Any:
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"Rate limited. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def run_once(query: str, backend: Backend) -> RunResult:
    chat_history: list[dict[str, Any]] = [{"role": "user", "content": query}]
    tool_call_count = 0

    result: StreamResult = call_with_backoff(lambda: backend.stream(chat_history))

    while result.stop_reason == "tool_use":
        tool_call_count += 1

        tool_fn = REGISTRY.get(result.tool_name)
        if tool_fn is None:
            tool_result = f"Unknown tool: {result.tool_name}"
        else:
            try:
                tool_result = call_with_backoff(lambda: tool_fn(**result.tool_input))
            except Exception as e:
                tool_result = f"Tool failed: {e}"

        chat_history.append(result.raw_content)
        chat_history.append(backend.tool_result_message(result.tool_id, tool_result))

        result = call_with_backoff(lambda: backend.stream(chat_history))

    return RunResult(response_text=result.text, tool_call_count=tool_call_count)


def run_agent(query: str, backend: Backend, memory_strategy: str = "full") -> None:
    # Inject persistent memory into the system prompt before the first turn
    memory_context = on_conversation_start(query)
    if memory_context:
        backend._system = backend._system + memory_context
        print("[persistent memory] Context loaded from previous sessions.")

    chat_history: list[dict[str, Any]] = [{"role": "user", "content": query}]

    while query != "exit":
        result: StreamResult = call_with_backoff(lambda: backend.stream(chat_history))

        while result.stop_reason == "tool_use":
            print(f"\nTool: {result.tool_name} | Input: {result.tool_input}")

            tool_fn = REGISTRY.get(result.tool_name)
            if tool_fn is None:
                tool_result = f"Unknown tool: {result.tool_name}"
            else:
                try:
                    tool_result = call_with_backoff(lambda: tool_fn(**result.tool_input))
                except Exception as e:
                    tool_result = f"Tool failed: {e}"

            chat_history.append(result.raw_content)
            chat_history.append(backend.tool_result_message(result.tool_id, tool_result))

            result = call_with_backoff(lambda: backend.stream(chat_history))

        print()
        report = verify_grounding(result.text, chat_history)
        print_grounding_report(report)

        chat_history.append(result.raw_content)
        chat_history, stats = apply_memory(chat_history, memory_strategy)
        print_memory_stats(stats)

        query = input("\n")
        if query == "exit":
            print("\n[persistent memory] Saving conversation...")
            stats = on_conversation_end(chat_history)
            print(
                f"[persistent memory] Extracted {stats['facts_extracted']} fact(s). "
                + (f"Episode #{stats['episode_id']} saved: \"{stats['summary_preview']}\"" if stats["episode_saved"] else "No episode saved.")
            )
            break
        chat_history.append({"role": "user", "content": query})
