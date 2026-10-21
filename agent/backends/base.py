from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class StreamResult:
    stop_reason: str
    raw_content: Any
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_id: str | None = None
    input_tokens: int = 0
    text: str = ""


class Backend(Protocol):
    def stream(self, messages: list[dict[str, Any]]) -> StreamResult: ...
    def tool_result_message(self, tool_id: str, content: str) -> dict[str, Any]: ...
