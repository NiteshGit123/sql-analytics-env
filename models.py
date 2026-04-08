"""
models.py
---------
Action / Observation / State typed models for the SQL Analytics environment.

The agent submits a SQL query (SQLAction) and receives back the query
result rows, any SQL error, the current task description, and reward
signal (SQLObservation).  Episode metadata is tracked in SQLState.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator

try:
    from openenv.core.env_server.interfaces import Action, Observation, State
except ImportError:
    from openenv.core.env_server.types import Action, Observation, State


class SQLAction(Action):
    """
    A single SQL query submitted by the agent.

    Example::

        SQLAction(sql="SELECT COUNT(*) FROM customers WHERE state = 'CA'")
    """

    sql: str = Field(
        ...,
        min_length=1,
        description="SQL SELECT query to execute against the e-commerce database.",
        json_schema_extra={"example": "SELECT COUNT(*) FROM customers WHERE state = 'CA'"},
    )

    @field_validator("sql")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        """Strip leading/trailing whitespace from the query."""
        return v.strip()


class SQLObservation(Observation):
    """
    Result of executing a SQL query in the environment.

    Fields
    ------
    result          : Rows returned by the query (list of dicts), None on error.
    error           : SQL error message if the query failed, else None.
    db_schema       : Human-readable database schema (populated on reset).
    task_description: Natural-language question the agent must answer.
    """

    result: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    db_schema: Optional[str] = None
    task_description: Optional[str] = None


class SQLState(State):
    """Episode-level state for the SQL Analytics environment."""

    task_id: str = ""
    task_description: str = ""
    difficulty: str = ""          # "easy" | "medium" | "hard"
    attempts: int = 0
    max_attempts: int = 10
    solved: bool = False
