"""
server/app.py
-------------
FastAPI application for the SQL Analytics Environment.

Exposes the environment over HTTP + WebSocket endpoints that are
consumed by EnvClient / the OpenEnv evaluation harness.

Usage (development):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

Usage (production via Docker):
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 1
"""

import time
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

try:
    from openenv.core.env_server.http_server import create_app
    from .sql_analytics_environment import SQLAnalyticsEnvironment
    from ..models import SQLAction, SQLObservation
except ImportError:
    from openenv.core.env_server.http_server import create_app
    from server.sql_analytics_environment import SQLAnalyticsEnvironment
    from models import SQLAction, SQLObservation

# ── Create base app ───────────────────────────────────────────────────────────
# Pass the class (not an instance) so the HTTP server can instantiate
# a fresh environment per WebSocket session.
app = create_app(
    SQLAnalyticsEnvironment,
    SQLAction,
    SQLObservation,
    env_name="sql_analytics_env",
)

# ── CORS — allow all origins (needed for HuggingFace Spaces) ─────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Explicit health check (required by OpenEnv validator + hackathon) ─────────
@app.get("/health", tags=["meta"])
async def health():
    """
    Health check endpoint.
    Returns HTTP 200 when the server is up and ready.
    Required by OpenEnv pre-submission checks.
    """
    return {"status": "ok", "environment": "sql_analytics_env", "version": "0.1.0"}


# ── Environment metadata ──────────────────────────────────────────────────────
@app.get("/info", tags=["meta"])
async def info():
    """Return environment metadata for discoverability."""
    return {
        "name":        "sql_analytics_env",
        "version":     "0.1.0",
        "description": (
            "SQL Analytics environment where AI agents answer "
            "business-intelligence questions by writing SQL queries "
            "against a seeded e-commerce database."
        ),
        "tasks": [
            {"task_id": "task_001", "difficulty": "easy",   "topic": "Count customers by state"},
            {"task_id": "task_002", "difficulty": "easy",   "topic": "Most expensive product in category"},
            {"task_id": "task_003", "difficulty": "medium", "topic": "Top 5 customers by spend"},
            {"task_id": "task_004", "difficulty": "medium", "topic": "Category with highest avg rating"},
            {"task_id": "task_005", "difficulty": "hard",   "topic": "Q1-only buyers (churn signal)"},
            {"task_id": "task_006", "difficulty": "hard",   "topic": "Monthly revenue + growth rate"},
        ],
        "action_space":      "SQLAction { sql: str }",
        "observation_space": "SQLObservation { result, error, db_schema, task_description, reward, done }",
        "reward_range":      [0.0, 1.0],
        "max_steps":         10,
    }


# ── Request timing middleware (helps with debugging slow queries) ─────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = round(time.perf_counter() - start, 4)
    response.headers["X-Process-Time"] = str(elapsed)
    return response


# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": type(exc).__name__,
            "detail": str(exc),
            "path": str(request.url),
        },
    )


def main() -> None:
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
