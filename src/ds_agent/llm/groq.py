from __future__ import annotations

import json
import os

import groq as _groq


def _to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
    """Convert Anthropic-style message history to OpenAI/Groq format."""
    result: list[dict] = [{"role": "system", "content": system}]

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            if isinstance(content, str):
                result.append({"role": "user", "content": content})
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "tool_result":
                        result.append({
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "content": (
                                block["content"]
                                if isinstance(block["content"], str)
                                else json.dumps(block["content"], default=str)
                            ),
                        })
                    elif block.get("type") == "text":
                        result.append({"role": "user", "content": block["text"]})

        elif role == "assistant":
            if isinstance(content, str):
                result.append({"role": "assistant", "content": content})
            elif isinstance(content, list):
                text = " ".join(
                    b["text"] for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
                tool_calls = [
                    {
                        "id": b["id"],
                        "type": "function",
                        "function": {
                            "name": b["name"],
                            "arguments": json.dumps(b.get("input") or {}, default=str),
                        },
                    }
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "tool_use"
                ]
                assistant_msg: dict = {"role": "assistant", "content": text or None}
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                result.append(assistant_msg)

    return result


def _to_openai_tools(tools: list[dict]) -> list[dict]:
    """Convert Anthropic tool definitions to OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for t in tools
    ]


def _to_anthropic_response(message) -> dict:
    """Convert a Groq/OpenAI chat completion message to Anthropic-style dict."""
    content: list[dict] = []

    if message.content:
        content.append({"type": "text", "text": message.content})

    for tc in message.tool_calls or []:
        try:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
        except json.JSONDecodeError:
            args = {}
        content.append({
            "type": "tool_use",
            "id": tc.id,
            "name": tc.function.name,
            "input": args,
        })

    return {"role": "assistant", "content": content}


class GroqClient:
    """LLM client backed by Groq's hosted inference API.

    Uses the GROQ_API_KEY environment variable (loaded from .env by the CLI).
    Translates between the agent's internal Anthropic message format and Groq's
    OpenAI-compatible API so the graph and tools are unaffected.
    """

    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        self._client = _groq.Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
        self._model = model

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> dict:
        openai_messages = _to_openai_messages(system, messages)
        openai_tools = _to_openai_tools(tools) if tools else None

        kwargs: dict = {"model": self._model, "messages": openai_messages}
        if openai_tools:
            kwargs["tools"] = openai_tools
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        return _to_anthropic_response(response.choices[0].message)
