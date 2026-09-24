from typing import TypedDict, List, Dict, Any


class InvestigationState(TypedDict, total=False):
    mission: str
    queries: List[str]
    evidence: List[Dict[str, Any]]
    findings: str
    verification: str
    needs_more: bool
    attempt: int
    final_report: str
