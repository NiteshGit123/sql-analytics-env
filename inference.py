"""
inference.py
------------
Baseline LLM agent for the SQL Analytics OpenEnv environment.

MANDATORY ENVIRONMENT VARIABLES
--------------------------------
    API_BASE_URL      The API endpoint for the LLM.
    MODEL_NAME        The model identifier to use for inference.
    HF_TOKEN          Your HuggingFace / API key.
    LOCAL_IMAGE_NAME  Local Docker image name (only if using from_docker_image())

STDOUT FORMAT (required by evaluator)
--------------------------------------
    [START] task=<task_id> env=sql_analytics_env model=<model_name>
    [STEP]  step=<n> action=<sql> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>

Rules:
    - One [START] per episode, one [STEP] per step, one [END] per episode.
    - reward and rewards formatted to 2 decimal places.
    - score formatted to 3 decimal places.
    - done and success are lowercase: true or false.
    - error is the raw error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - [END] is always emitted, even on exception.
"""

from __future__ import annotations

import os
import sys
import time
from typing import List, Optional

from openai import OpenAI, RateLimitError, APIConnectionError, APIStatusError

# ── Environment client ────────────────────────────────────────────────────────
try:
    from client import SQLAnalyticsEnv
    from models import SQLAction
except ImportError:
    from sql_analytics_env.client import SQLAnalyticsEnv
    from sql_analytics_env.models import SQLAction

# ── Config ────────────────────────────────────────────────────────────────────

API_KEY          = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL     = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME       = os.getenv("MODEL_NAME")   or "Qwen/Qwen2.5-72B-Instruct"
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")   # only needed for from_docker_image()
ENV_BASE_URL     = os.getenv("ENV_BASE_URL",  "http://localhost:8000")
MAX_STEPS        = int(os.getenv("MAX_STEPS", "8"))
BENCHMARK        = "sql_analytics_env"

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

SUCCESS_SCORE_THRESHOLD = 0.8  # reward >= 0.8 counts as solved

# ── Logging helpers (required stdout format) ──────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    # Escape newlines in SQL so it stays on one line
    action_clean = action.replace("\n", " ").replace("\r", "")
    error_val = error.replace("\n", " ") if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM prompt ────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert SQL analyst. You will be given a business
intelligence question about an e-commerce database and must answer it by
writing a single SQL query.

Rules:
- Write ONLY the raw SQL query — no markdown, no explanation, no backticks.
- The database is SQLite; use SQLite-compatible syntax.
- Use ROUND(..., 2) for monetary values and ratings.
- Use strftime('%m', date_column) to extract the month as a string.
- When asked for growth rate, use a self-join or subquery (SQLite does not
  always support window function LAG).
- Return only the columns asked for in the task description.
"""


def _build_user_message(schema: str, task_description: str, history: list) -> str:
    lines = ["DATABASE SCHEMA:", schema, "", "TASK:", task_description]
    if history:
        lines += ["", "PREVIOUS ATTEMPTS (most recent first):"]
        for h in reversed(history[-3:]):
            lines.append(f"  SQL: {h['sql']}")
            if h.get("error"):
                lines.append(f"  Error: {h['error']}")
            else:
                lines.append(
                    f"  Rows: {h.get('result_rows', 0)} | Reward: {h.get('reward', 0.0):.2f}"
                )
        lines += ["", "Please write an improved SQL query."]
    else:
        lines += ["", "Write the SQL query:"]
    return "\n".join(lines)


def _extract_sql(text: str) -> str:
    sql = text.strip()
    for fence in ("```sql", "```SQL", "```"):
        if sql.startswith(fence):
            sql = sql[len(fence):]
            if "```" in sql:
                sql = sql[: sql.index("```")]
            break
    return sql.strip()


# ── Agent loop ────────────────────────────────────────────────────────────────

def run_task(client: OpenAI, env: SQLAnalyticsEnv, task_id: str) -> float:
    """Run one episode. Returns the final score in [0, 1]."""
    obs = env.reset(task_id=task_id)
    schema           = obs.db_schema or ""
    task_description = obs.task_description or ""

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    history: list = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        for step in range(1, MAX_STEPS + 1):
            user_msg = _build_user_message(schema, task_description, history)
            sql = ""

            # LLM call with retry + exponential backoff
            for attempt in range(1, 4):
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
                    break
                except RateLimitError:
                    time.sleep(2 ** attempt)
                except APIConnectionError:
                    time.sleep(2)
                except (APIStatusError, Exception) as exc:
                    print(f"[DEBUG] LLM error step={step}: {exc}", file=sys.stderr, flush=True)
                    break

            if not sql:
                print(f"[DEBUG] No SQL at step={step}, stopping.", file=sys.stderr, flush=True)
                break

            obs = env.step(SQLAction(sql=sql))

            reward      = obs.reward or 0.0
            done        = obs.done
            error       = obs.error
            result_rows = len(obs.result) if obs.result else 0
            steps_taken = step

            rewards.append(reward)
            log_step(step=step, action=sql, reward=reward, done=done, error=error)

            history.append({
                "sql":         sql,
                "error":       error,
                "result_rows": result_rows,
                "reward":      reward,
            })

            if done:
                break

        score   = rewards[-1] if rewards else 0.0
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Unhandled exception in run_task: {exc}", file=sys.stderr, flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env    = SQLAnalyticsEnv(base_url=ENV_BASE_URL)

    scores = {}
    try:
        for task_id in ALL_TASK_IDS:
            scores[task_id] = run_task(client, env, task_id)
    finally:
        env.close()

    avg = sum(scores.values()) / len(scores) if scores else 0.0
    solved = sum(1 for s in scores.values() if s >= SUCCESS_SCORE_THRESHOLD)
    print(
        f"[SUMMARY] tasks={len(scores)} solved={solved} avg_score={avg:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
