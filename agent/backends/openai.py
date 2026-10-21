import json
from openai import OpenAI
from typing import Any

from config import SYSTEM
from tools import TOOL_SCHEMAS
from .base import StreamResult


def _to_openai_tools(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": s["input_schema"],
            },
        }
        for s in schemas
    ]


class OpenAIBackend:
    def __init__(self, model: str, base_url: str | None = None, system: str | None = None):
        self.model = model
        self._system = system or SYSTEM
        self._tools = _to_openai_tools(TOOL_SCHEMAS)
        if base_url:
            self._client = OpenAI(base_url=base_url, api_key="ollama")
        else:
            self._client = OpenAI()

    def stream(self, messages: list[dict[str, Any]], on_token=None) -> StreamResult:
        full_messages = [{"role": "system", "content": self._system}] + messages
        response = self._client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            tools=self._tools,
            tool_choice="auto",
            stream=True,
            stream_options={"include_usage": True},
        )

        text = ""
        tool_chunks: dict[int, dict[str, str]] = {}
        finish_reason = None
        input_tokens = 0

        for chunk in response:
            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            finish_reason = choice.finish_reason or finish_reason
            delta = choice.delta

            if delta.content:
                if on_token:
                    on_token(delta.content)
                else:
                    print(delta.content, end="", flush=True)
                text += delta.content

            for tc in delta.tool_calls or []:
                entry = tool_chunks.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                if tc.id:
                    entry["id"] = tc.id
                if tc.function.name:
                    entry["name"] = tc.function.name
                if tc.function.arguments:
                    entry["arguments"] += tc.function.arguments

        raw_content: dict[str, Any] = {"role": "assistant", "content": text or None}
        tool_name = tool_input = tool_id = None

        if finish_reason == "tool_calls" and tool_chunks:
            tc = tool_chunks[0]
            tool_name = tc["name"]
            tool_input = json.loads(tc["arguments"])
            tool_id = tc["id"]
            raw_content["tool_calls"] = [
                {
                    "id": t["id"],
                    "type": "function",
                    "function": {"name": t["name"], "arguments": t["arguments"]},
                }
                for t in tool_chunks.values()
            ]

        return StreamResult(
            stop_reason="tool_use" if finish_reason == "tool_calls" else "end_turn",
            raw_content=raw_content,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_id=tool_id,
            input_tokens=input_tokens,
            text=text,
        )

    def tool_result_message(self, tool_id: str, content: Any) -> dict[str, Any]:
        if not isinstance(content, str):
            content = json.dumps(content)
        return {"role": "tool", "tool_call_id": tool_id, "content": content}
