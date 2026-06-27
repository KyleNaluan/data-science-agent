from abc import ABC, abstractmethod

import pandas as pd


class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def input_schema(self) -> dict: ...

    @abstractmethod
    def run(self, df: pd.DataFrame, **kwargs) -> dict:
        """Execute tool on a DataFrame.

        MUST return aggregate-only data (column-level stats, counts, schema).
        Must never include a row that ties multiple field values together.
        See ADR-0002.
        """
        ...

    def to_anthropic_def(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
