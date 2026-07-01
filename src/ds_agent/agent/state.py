import operator
from typing import Annotated, Any

from typing_extensions import TypedDict


class AgentState(TypedDict):
    csv_path: str
    output_dir: str
    df: Any  # pd.DataFrame — lives in memory only, never sent to LLM
    messages: Annotated[list[dict], operator.add]
    tool_results: dict  # {tool_name: tool_output_dict}
    trace_entries: Annotated[list[dict], operator.add]
    report: str
    flagged_assumptions: Annotated[list[dict], operator.add]
    extra_csv_paths: list[str]  # additional CSVs for multi-CSV join (issue #8)
