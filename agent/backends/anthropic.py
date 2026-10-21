import json
from anthropic import Anthropic
from typing import Any

from config import SYSTEM
from tools import TOOL_SCHEMAS
from .base import StreamResult


class AnthropicBackend:
    def __init__(self, model: str, system: str | None = None):
        self.model = model
        self._system = system or SYSTEM
        self._client = Anthropic()

    def stream(self, messages: list[dict[str, Any]], on_token=None) -> StreamResult:
        with self._client.messages.stream(
            model=self.model,
            max_tokens=1024,
            system=self._system,
            tools=TOOL_SCHEMAS,
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
            messages=messages,
        ) as s:
            for text in s.text_stream:
                if on_token:
                    on_token(text)
                else:
                    print(text, end="", flush=True)
            response = s.get_final_message()

        tool_name = tool_input = tool_id = None
        if response.stop_reason == "tool_use":
            block = next(b for b in response.content if b.type == "tool_use")
            tool_name, tool_input, tool_id = block.name, block.input, block.id

        text = "".join(b.text for b in response.content if b.type == "text")

        return StreamResult(
            stop_reason=response.stop_reason,
            raw_content={"role": "assistant", "content": response.content},
            tool_name=tool_name,
            tool_input=tool_input,
            tool_id=tool_id,
            input_tokens=response.usage.input_tokens,
            text=text,
        )

    def tool_result_message(self, tool_id: str, content: Any) -> dict[str, Any]:
        if not isinstance(content, str):
            content = json.dumps(content)
        return {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": content}],
        }
