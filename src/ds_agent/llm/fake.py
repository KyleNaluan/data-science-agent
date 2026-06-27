class FakeLLMClient:
    """Scripted LLM client for integration tests.

    Replays a fixed list of responses in order. Each response must be an
    Anthropic-style assistant message dict.
    """

    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self._index = 0

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> dict:
        if self._index >= len(self._responses):
            raise ValueError(
                f"FakeLLMClient exhausted its scripted responses after {self._index} call(s). "
                "Add more responses or check your graph loop termination."
            )
        response = self._responses[self._index]
        self._index += 1
        return response

    @property
    def call_count(self) -> int:
        return self._index
