import anthropic as _anthropic


class AnthropicClient:
    def __init__(self, model: str = "claude-sonnet-4-6", **client_kwargs):
        self._client = _anthropic.Anthropic(**client_kwargs)
        self._model = model

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> dict:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system,
            messages=messages,
            tools=tools or _anthropic.NOT_GIVEN,
        )
        return {
            "role": "assistant",
            "content": [block.model_dump() for block in response.content],
        }
