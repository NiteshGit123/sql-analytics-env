"""
inference.py
------------
Baseline LLM agent for the SQL Analytics OpenEnv environment.

The agent receives a natural-language business-intelligence question,
iteratively generates SQL queries using an LLM (via OpenAI-compatible
API), executes them in the environment, and improves based on feedback.

Logging
-------
Every event is printed to stdout in the strictly-required format:

    [START] {"task_id": ..., "task_description": ..., "difficulty": ...}
    [STEP]  {"task_id": ..., "step": ..., "sql": ..., "reward": ...,
             "result_rows": ..., "error": ..., "done": ...}
    [END]   {"task_id": ..., "final_score": ..., "total_steps": ...,
             "success": ..., "timestamp": ...}

Environment variables
---------------------
    API_BASE_URL   Base URL for the OpenAI-compatible LLM endpoint
                   (default: https://api.openai.com/v1)
    MODEL_NAME     LLM model to use (default: gpt-4o-mini)
    HF_TOKEN       HuggingFace token when connecting to HF Spaces
    ENV_BASE_URL   URL of the running sql_analytics_env server
                   (default: http://localhost:8000)
    MAX_STEPS      Max LLM turns per task (default: 8)

Usage
-----
    # Against local server started with:  uvicorn server.app:app --port 8000
    python inference.py

    # Against a HuggingFace Space:
    ENV_BASE_URL=https://<your-space>.hf.space python inference.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from openai import OpenAI, RateLimitError, APIConnectionError, APIStatusError

# ── Environment client ────────────────────────────────────────────────────────
# Support running inference.py both as standalone and inside the package.
try:
    from client import SQLAnalyticsEnv
    from models import SQLAction
except ImportError:
    from sql_analytics_env.client import SQLAnalyticsEnv
    from sql_analytics_env.models import SQLAction

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE_URL     = os.environ.get("API_BASE_URL",  "https://api.openai.com/v1")
MODEL_NAME       = os.environ.get("MODEL_NAME",    "gpt-4o-mini")
HF_TOKEN         = os.environ.get("HF_TOKEN")           # no default — must be set externally
LOCAL_IMAGE_NAME = os.environ.get("LOCAL_IMAGE_NAME")   # optional, only for from_docker_image()
ENV_BASE_URL     = os.environ.get("ENV_BASE_URL",  "http://localhost:8000")
MAX_STEPS        = int(os.environ.get("MAX_STEPS", "8"))

ALL_TASK_IDS = [
    "task_001",  # easy
    "task_002",  # easy
    "task_003",  # medium
    "task_004",  # medium
    "task_005",  # hard
    "task_006",  # hard
    "task_007",  # easy
    "task_008",  # easy
    "task_009",  # medium
    "task_010",  # medium
]

# ── Logging helpers ───────────────────────────────────────────────────────────

def _emit(tag: str, payload: Dict[str, Any]) -> None:
    """Print a structured log line to stdout (required evaluation format)."""
    print(f"[{tag}] {json.dumps(payload, default=str)}", flush=True)


def log_start(task_id: str, task_description: str, difficulty: str) -> None:
    _emit("START", {
        "task_id":          task_id,
        "task_description": task_description,
        "difficulty":       difficulty,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
    })


def log_step(
    task_id: str,
    step:    int,
    sql:     str,
    reward:  float,
    result_rows: int,
    error:   Optional[str],
    done:    bool,
) -> None:
    _emit("STEP", {
        "task_id":     task_id,
        "step":        step,
        "sql":         sql,
        "reward":      reward,
        "result_rows": result_rows,
        "error":       error,
        "done":        done,
    })


def log_end(
    task_id:     str,
    final_score: float,
    total_steps: int,
    success:     bool,
) -> None:
    _emit("END", {
        "task_id":     task_id,
        "final_score": final_score,
        "total_steps": total_steps,
        "success":     success,
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    })


# ── LLM helpers ───────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert SQL analyst. You will be given a business
intelligence question about an e-commerce database and must answer it by
writing a single SQL query.

Rules:
- Write ONLY the raw SQL query — no markdown, no explanation, no backticks.
- The database is SQLite; use SQLite-compatible syntax.
- Use ROUND(..., 2) for monetary values and ratings.
- Use strftime('%m', date_column) to extract the month as a string;
  CAST(strftime('%m', ...) AS INTEGER) for an integer.
- When asked for growth rate you may use a self-join or a subquery
  (SQLite does not always support window functions LAG).
- Return only the columns asked for in the task description.
"""

def _build_user_message(
    schema:           str,
    task_description: str,
    history:          List[Dict],
) -> str:
    """Build the user turn including schema, task, and any prior attempt feedback."""
    lines = [
        "DATABASE SCHEMA:",
        schema,
        "",
        "TASK:",
        task_description,
    ]

    if history:
        lines += ["", "PREVIOUS ATTEMPTS (most recent first):"]
        for attempt in reversed(history[-3:]):   # show last 3
            lines.append(f"  SQL: {attempt['sql']}")
            if attempt.get("error"):
                lines.append(f"  Error: {attempt['error']}")
            else:
                rows = attempt.get("result_rows", 0)
                reward = attempt.get("reward", 0.0)
                msg = attempt.get("message", "")
                lines.append(f"  Rows returned: {rows} | Reward: {reward:.2f} | Feedback: {msg}")
        lines += ["", "Please write an improved SQL query."]
    else:
        lines += ["", "Write the SQL query:"]

    return "\n".join(lines)


def _extract_sql(llm_response: str) -> str:
    """Strip markdown fences and whitespace from the LLM output."""
    sql = llm_response.strip()
    for fence in ("```sql", "```SQL", "```"):
        if sql.startswith(fence):
            sql = sql[len(fence):]
            if "```" in sql:
                sql = sql[: sql.index("```")]
            break
    return sql.strip()


# ── Agent loop ────────────────────────────────────────────────────────────────

def run_task(
    client:  OpenAI,
    env:     SQLAnalyticsEnv,
    task_id: str,
) -> float:
    """Run one task episode. Returns the final score (0.0–1.0)."""
    # Reset
    obs = env.reset(task_id=task_id)
    schema           = obs.db_schema or ""
    task_description = obs.task_description or ""
    difficulty       = obs.metadata.get("difficulty", "?") if obs.metadata else "?"

    log_start(task_id, task_description, difficulty)

    history: List[Dict] = []
    final_score = 0.0
    step_count  = 0

    for step in range(1, MAX_STEPS + 1):
        # Ask LLM (with retry + exponential backoff)
        user_msg = _build_user_message(schema, task_description, history)
        sql = ""
        for attempt in range(1, 4):   # up to 3 retries
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user",   "content": user_msg},
                    ],
                    temperature=0.0,
                    max_tokens=512,
                )
                sql = _extract_sql(response.choices[0].message.content or "")
                break   # success
            except RateLimitError:
                wait = 2 ** attempt
                print(f"[WARN] Rate limited — waiting {wait}s (attempt {attempt}/3)", file=sys.stderr)
                time.sleep(wait)
            except APIConnectionError as exc:
                print(f"[WARN] Connection error on step {step}, attempt {attempt}: {exc}", file=sys.stderr)
                time.sleep(2)
            except APIStatusError as exc:
                print(f"[WARN] API error {exc.status_code} on step {step}: {exc.message}", file=sys.stderr)
                break
            except Exception as exc:
                print(f"[WARN] Unexpected LLM error on step {step}: {exc}", file=sys.stderr)
                break

        if not sql:
            print(f"[WARN] No SQL generated for step {step} — skipping.", file=sys.stderr)
            break

        if not sql:
            break

        # Execute in environment
        obs = env.step(SQLAction(sql=sql))

        result_rows = len(obs.result) if obs.result else 0
        reward      = obs.reward or 0.0
        error       = obs.error
        done        = obs.done
        message     = (obs.metadata or {}).get("message", "")
        step_count  = step

        log_step(task_id, step, sql, reward, result_rows, error, done)

        history.append({
            "sql":         sql,
            "error":       error,
            "result_rows": result_rows,
            "reward":      reward,
            "message":     message,
        })

        final_score = reward

        if done:
            break

    success = final_score >= 0.9
    log_end(task_id, final_score, step_count, success)
    return final_score


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    # Build OpenAI client — uses API_BASE_URL and MODEL_NAME from env vars
    llm_client = OpenAI(base_url=API_BASE_URL)

    # HF_TOKEN is read from env automatically by the OpenEnv framework
    # when connecting to a HuggingFace Space
    env = SQLAnalyticsEnv(base_url=ENV_BASE_URL)

    scores: Dict[str, float] = {}

    try:
        for task_id in ALL_TASK_IDS:
            score = run_task(llm_client, env, task_id)
            scores[task_id] = score
    finally:
        env.close()

    # Summary
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(
        json.dumps({
            "summary": {
                "scores":        scores,
                "average_score": round(avg, 4),
                "tasks_solved":  sum(1 for s in scores.values() if s >= 0.9),
                "total_tasks":   len(scores),
            }
        }),
        flush=True,
    )


if __name__ == "__main__":
    main()
