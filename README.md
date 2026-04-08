# SQL Analytics OpenEnv Environment

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![OpenEnv](https://img.shields.io/badge/OpenEnv-spec__v1-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Tasks](https://img.shields.io/badge/Tasks-6%20(easy%2Fmedium%2Fhard)-orange)

A real-world OpenEnv environment where an AI agent answers business-intelligence
questions by writing SQL queries against a seeded e-commerce database.

---

## Overview

The agent interacts with a **SQLite e-commerce database** containing:

| Table        | Rows  | Description                                  |
|--------------|-------|----------------------------------------------|
| customers    |  100  | Registered users with US state + signup date |
| categories   |    5  | Electronics, Clothing, Books, Food, Sports   |
| products     |   50  | Name, category, price, stock                 |
| orders       |  500  | Date, total amount, status (delivered/…)     |
| order_items  | 1000+ | Per-line quantity and unit price             |
| reviews      |  300  | 1–5 star ratings per product                 |

---

## Database Schema

```
customers(id, name, email, state, signup_date, last_order_date)
categories(id, name)
products(id, name, category_id, price, stock)
orders(id, customer_id, order_date, total_amount, status)
order_items(id, order_id, product_id, quantity, unit_price)
reviews(id, product_id, customer_id, rating, review_date)
```

---

## Tasks

| Task ID   | Difficulty | Question                                              |
|-----------|------------|-------------------------------------------------------|
| task_001  | Easy       | Count customers from California                       |
| task_002  | Easy       | Name + price of the most expensive Electronics item   |
| task_003  | Medium     | Top 5 customers by total spend                        |
| task_004  | Medium     | Product category with highest average rating          |
| task_005  | Hard       | Q1-2024-only buyers (early-churn signal)              |
| task_006  | Hard       | Monthly 2024 revenue + month-over-month growth rate   |

---

## Action / Observation Spaces

### Action — `SQLAction`

| Field | Type | Description                           |
|-------|------|---------------------------------------|
| `sql` | str  | SQL SELECT query (SQLite dialect)     |

### Observation — `SQLObservation`

| Field              | Type               | Description                              |
|--------------------|--------------------|------------------------------------------|
| `result`           | `list[dict] / None`| Query result rows                        |
| `error`            | `str / None`       | SQL error message (if any)               |
| `db_schema`        | `str / None`       | Full schema description (on reset only)  |
| `task_description` | `str / None`       | Natural-language task question           |
| `reward`           | `float`            | Partial-progress reward  [0.0 – 1.0]    |
| `done`             | `bool`             | True when episode ends                   |
| `metadata`         | `dict`             | task_id, difficulty, attempts, message   |

### Reward Signal (partial progress)

| Condition                            | Reward    |
|--------------------------------------|-----------|
| SQL syntax / write-op error          | 0.0       |
| Valid SQL but wrong answer           | 0.1 – 0.5 |
| Near-correct (close count / overlap) | 0.5 – 0.9 |
| Correct answer                       | 1.0       |

### Episode ends (`done=True`) when:
1. Agent submits a correct answer (`reward >= 0.9`)
2. Agent exhausts 10 attempts without solving
3. `step()` called before `reset()`

---

## Installation

### Prerequisites
- Python 3.10+
- pip
- (Optional) Docker for containerized deployment

### Setup

```bash
# Clone / navigate to the environment folder
cd sql_analytics_env

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux

# Install dependencies
pip install openenv-core fastapi "uvicorn[standard]" pydantic "openai>=1.30.0"
```

---

## Running Locally

### Start the server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Verify health

```bash
curl.exe http://localhost:8000/health
# {"status": "ok", "environment": "sql_analytics_env", "version": "0.1.0"}
```

### Browse API docs

Open in browser: **http://localhost:8000/docs** (Swagger UI)

### Run smoke tests

```bash
python -X utf8 test_env.py
```

Expected: all `[PASS]`, Average correct reward = 1.00

---

## Docker

```bash
# Build
docker build -t sql-analytics-env .

# Run
docker run -p 8000:8000 sql-analytics-env

# Verify
curl http://localhost:8000/health
```

---

## Running the Baseline Agent

```bash
# Set LLM credentials
set OPENAI_API_KEY=sk-your-key-here        # Windows CMD
# export OPENAI_API_KEY=sk-your-key-here  # Mac/Linux / PowerShell

# Optional overrides
set MODEL_NAME=gpt-4o-mini
set ENV_BASE_URL=http://localhost:8000
set MAX_STEPS=8

# Run inference
python -X utf8 inference.py
```

### Expected output

```
[START] {"task_id": "task_001", "task_description": "...", "difficulty": "easy", ...}
[STEP]  {"task_id": "task_001", "step": 1, "sql": "SELECT COUNT(*) ...", "reward": 1.0, ...}
[END]   {"task_id": "task_001", "final_score": 1.0, "total_steps": 1, "success": true, ...}
...
{"summary": {"scores": {...}, "average_score": 0.95, "tasks_solved": 6, "total_tasks": 6}}
```

Runs in well under **20 minutes** on a 2 vCPU / 8 GB machine.

---

## HuggingFace Spaces Deployment

1. Create a new Space → **Docker** SDK type
2. Push this folder as the Space repository root
3. HF Spaces builds from `Dockerfile` and serves on port 8000
4. Set `ENV_BASE_URL=https://<your-space>.hf.space` in your inference runner

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `uvicorn: command not found` | Activate venv: `.venv\Scripts\activate` |
| `Port 8000 already in use` | Use `--port 8001` |
| `ModuleNotFoundError: openenv` | Run `pip install openenv-core` |
| Unicode error on Windows | Always use `python -X utf8 inference.py` |
| `curl` not working in PowerShell | Use `curl.exe` or `Invoke-WebRequest -UseBasicParsing` |
| LLM rate limit in inference.py | Agent auto-retries with exponential backoff |
| SQL query times out | Queries limited to 5 seconds; simplify your query |

---

## Pre-submission Checklist

- [x] HF Space returns HTTP 200 on `/health`
- [x] `openenv.yaml` passes `openenv validate`
- [x] `Dockerfile` builds successfully with HEALTHCHECK
- [x] `inference.py` in root directory
- [x] Baseline script runs in < 20 minutes
- [x] Structured `[START]` / `[STEP]` / `[END]` stdout logging
- [x] 6 tasks with easy / medium / hard graders
- [x] Partial-progress reward functions (0.0 to 1.0)
- [x] Compatible with 2 vCPU / 8 GB deployment
- [x] CORS enabled for cross-origin access
- [x] Query timeout (5s) prevents hangs
- [x] Write-operation blocking (SELECT only)
- [x] Retry logic in inference.py for LLM failures
- [x] `.dockerignore` excludes venv and cache files
