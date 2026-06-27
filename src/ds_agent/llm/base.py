from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    def complete(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
    ) -> dict:
        """
        Make an LLM call with the given system prompt, message history, and tool definitions.

        Returns an Anthropic-style assistant message:
            {"role": "assistant", "content": [<content blocks>]}

        Content blocks are dicts with "type" == "text" or "type" == "tool_use".
        A "tool_use" block has: {"type": "tool_use", "id": str, "name": str, "input": dict}
        A "text" block has: {"type": "text", "text": str}
        """
        ...
