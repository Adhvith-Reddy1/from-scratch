import re
import json
from anthropic import Anthropic

_client = Anthropic()

JUDGE_PROMPT = """You are a fact-checker. Determine whether the source passage supports the given claim.

Claim: {claim}

Source passage:
\"\"\"{passage}\"\"\"

Does this passage directly support the claim?
Answer with exactly one word on the first line: SUPPORTED, PARTIALLY, or UNSUPPORTED
Then on the next line, one sentence explaining why."""


def extract_claim_source_pairs(text: str) -> list[tuple[str, str]]:
    """Extract (claim, source_ref) pairs from [Source: ...] citations in agent response."""
    pattern = r'([^.\n!?]+(?:[.!?])?)\s*\[Source:\s*([^\]]+)\]'
    return [(m.group(1).strip(), m.group(2).strip()) for m in re.finditer(pattern, text)]


def _collect_tool_contents(chat_history: list[dict]) -> list[str]:
    """Pull raw tool result strings from chat history (handles both Anthropic and OpenAI formats)."""
    contents = []
    for msg in chat_history:
        role = msg.get("role")
        content = msg.get("content", "")

        # OpenAI/Ollama: role=tool, content=string
        if role == "tool" and isinstance(content, str):
            contents.append(content)

        # Anthropic: role=user, content=[{type: tool_result, content: string}]
        elif role == "user" and isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tc = block.get("content", "")
                    if isinstance(tc, str):
                        contents.append(tc)
    return contents


def _find_passage(source_ref: str, tool_contents: list[str]) -> str | None:
    """Match a citation source reference to a passage in the tool results."""
    source_key = source_ref.split(",")[0].strip()

    for raw in tool_contents:
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            parsed = raw

        if isinstance(parsed, list) and parsed:
            # Local KB: list of cited strings "text\n[Source: ...]"
            if isinstance(parsed[0], str):
                for item in parsed:
                    if source_key in item:
                        return item
            # Web search: list of {"title": ..., "url": ..., "content": ...}
            elif isinstance(parsed[0], dict):
                for item in parsed:
                    if source_key in item.get("url", "") or source_key in item.get("title", ""):
                        return item.get("content", "")

        elif isinstance(parsed, str) and source_key in parsed:
            return parsed

    return None


def _judge(claim: str, passage: str) -> dict:
    response = _client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=120,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(claim=claim, passage=passage[:2000])}],
    )
    lines = response.content[0].text.strip().split("\n", 1)
    verdict = lines[0].strip().upper()
    if verdict not in ("SUPPORTED", "PARTIALLY", "UNSUPPORTED"):
        verdict = "UNSUPPORTED"
    explanation = lines[1].strip() if len(lines) > 1 else ""
    return {"verdict": verdict, "explanation": explanation}


def verify_grounding(response_text: str, chat_history: list[dict]) -> dict:
    pairs = extract_claim_source_pairs(response_text)
    if not pairs:
        return {"claims_checked": 0, "results": []}

    tool_contents = _collect_tool_contents(chat_history)
    results = []

    for claim, source_ref in pairs:
        passage = _find_passage(source_ref, tool_contents)
        if passage is None:
            results.append({
                "claim": claim,
                "source": source_ref,
                "verdict": "SOURCE_NOT_FOUND",
                "explanation": "Cited source could not be matched to any retrieved context.",
            })
        else:
            judgment = _judge(claim, passage)
            results.append({"claim": claim, "source": source_ref, **judgment})

    counts = {"SUPPORTED": 0, "PARTIALLY": 0, "UNSUPPORTED": 0, "SOURCE_NOT_FOUND": 0}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1

    return {"claims_checked": len(results), "counts": counts, "results": results}


def print_grounding_report(report: dict) -> None:
    if report["claims_checked"] == 0:
        print("\n── Grounding: no citations found in response ──")
        return

    c = report["counts"]
    icons = {"SUPPORTED": "✅", "PARTIALLY": "⚠️ ", "UNSUPPORTED": "❌", "SOURCE_NOT_FOUND": "❓"}

    print(f"\n── Grounding Verification ({report['claims_checked']} claims) ──")
    print(f"   ✅ {c['SUPPORTED']}  ⚠️  {c['PARTIALLY']}  ❌ {c['UNSUPPORTED']}  ❓ {c['SOURCE_NOT_FOUND']}")

    for r in report["results"]:
        icon = icons.get(r["verdict"], "?")
        print(f"\n   {icon} \"{r['claim'][:90]}\"")
        print(f"      Source: {r['source']}")
        if r.get("explanation"):
            print(f"      {r['explanation']}")
