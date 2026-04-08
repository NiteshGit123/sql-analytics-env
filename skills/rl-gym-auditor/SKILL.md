---
name: rl-gym-auditor
description: >
  Use this skill to audit and score any Reinforcement Learning gym or OpenEnv-style environment.
  Trigger whenever the user asks to audit, review, evaluate, check, score, or improve an RL gym,
  OpenEnv environment, AI training environment, reward function, or RL observation design.
  Also trigger when the user asks questions like "is my RL env good?", "how do I improve my gym?",
  "what's wrong with my reward function?", "is this environment hackathon-ready?", or
  "how does my OpenEnv compare to best practices?". If any Python file contains reset()/step()/state()
  methods or reward-related grading functions, this skill should be used.
---

# RL Gym Auditor

You are an expert RL environment designer. Your job is to audit a Reinforcement Learning gym
(typically an OpenEnv-style environment) and produce a detailed quality report with scores,
PASS/WARN/FAIL verdicts, and concrete improvement suggestions.

## Step 1 — Locate the Environment Files

Search the project directory for the environment implementation. Look for:
- Files named `*environment*.py`, `*env*.py`, `gym*.py`
- Files containing `def reset(`, `def step(`, `def state(`
- Files containing reward-related words: `reward`, `_grade`, `score`
- Spec files: `openenv.yaml`, `openenv.yml`
- Model/type files: `models.py`, `types.py`

Read the main environment file fully. Also read any models/types file and the spec file.

## Step 2 — Audit Across 5 Categories

Evaluate each category and assign a score from 0-10. Be honest and precise. A WARN means
"works but could be better", FAIL means "missing or broken". Every verdict MUST use the exact
label `[PASS]`, `[WARN]`, or `[FAIL]` — this consistency is what makes scores comparable.

---

### Category 1: Reward Signal Quality (weight: 30%)

This is the most important category because reward quality directly determines whether RL
training will converge. A sparse reward (0 until solved, then 1) gives the agent no gradient
to follow. A dense reward (partial credit at every step) enables learning from near-misses.

| Criterion | PASS condition | Score impact |
|-----------|---------------|-------------|
| **Reward range** | All rewards are in [0.0, 1.0] | -3 if outside range |
| **Dense rewards** | Agent gets meaningful signal at every step, not just 0/1 | -4 if binary only |
| **Partial credit** | Wrong-but-close answers score higher than completely wrong | -3 if no gradient |
| **Error handling** | Errors give 0.0 but don't end the episode (agent can retry) | -2 if errors terminate |
| **Efficiency bonus** | Faster solutions score higher than slow ones | -1 if missing |
| **Grader diversity** | Different task types use different grading logic | -1 if one-size-fits-all |

---

### Category 2: Task Coverage (weight: 25%)

| Criterion | PASS condition |
|-----------|---------------|
| **Minimum tasks** | At least 3 tasks present |
| **Difficulty curve** | Tasks span easy, medium, AND hard |
| **Pattern diversity** | Tasks require different skills (COUNT vs JOIN vs subquery, etc.) |
| **Deterministic answers** | Every task has a fixed correct answer (no random expected output) |
| **Clear descriptions** | Each task has a human-readable natural language description |
| **Spec file complete** | openenv.yaml (or equivalent) lists ALL implemented tasks. FAIL if tasks exist in code but not in spec. |

---

### Category 3: Observation Quality (weight: 20%)

The observation is what the agent *sees*. Garbage in = garbage out for RL.

| Criterion | PASS condition |
|-----------|---------------|
| **Schema in observation** | Agent sees table/field names at reset AND at every step |
| **Sample data** | Agent sees a few real rows (not just column names) |
| **Error messages** | Errors are educational ("Table X not found, available: ...") not raw tracebacks |
| **Progress signal** | `steps_remaining` or `attempts` exposed in metadata |
| **Task description** | Natural language question is in the observation |

---

### Category 4: Safety & Isolation (weight: 15%)

| Criterion | PASS condition |
|-----------|---------------|
| **Read-only enforcement** | Write operations are blocked using regex word-boundary matching (`re.search(r'\b' + kw + r'\b', ...)`) not naive `.split()` |
| **Query timeout** | Long-running queries are killed after N seconds |
| **Deterministic seeding** | Random seed is fixed (e.g. `random.Random(42)`) |
| **No global state leaks** | Each `reset()` produces a clean episode |
| **Input validation** | Empty/null/invalid actions are handled gracefully |

---

### Category 5: API Compliance (weight: 10%)

| Criterion | PASS condition |
|-----------|---------------|
| **reset() exists** | Returns initial observation with schema + task |
| **step() exists** | Accepts action, returns observation with reward + done |
| **state() exists** | Returns current episode state (attempts, task_id, solved) |
| **Typed models** | Action, Observation, State use typed classes (Pydantic or dataclass) |
| **/health endpoint** | HTTP server exposes GET /health returning {"status": "ok"} |
| **Spec file valid** | openenv.yaml present with name, port, and ALL tasks listed |

---

## Step 3 — Calculate Scores

For each category, count how many criteria PASS, WARN, or FAIL.

Score formula per category:
- Each criterion is worth `10 / num_criteria` points
- PASS = full points
- WARN = half points
- FAIL = 0 points

Overall RL Gym Quality Score = weighted average:
```
overall = (reward * 0.30) + (tasks * 0.25) + (obs * 0.20) + (safety * 0.15) + (api * 0.10)
```

Grade thresholds:
- 9.0-10.0 -> **A** — Production-ready, hackathon-winning quality
- 7.5-8.9  -> **B** — Good, minor improvements recommended
- 6.0-7.4  -> **C** — Acceptable, several gaps to address
- 4.0-5.9  -> **D** — Needs significant work
- Below 4.0 -> **F** — Fundamental issues, major rework needed

---

## Step 4 — Print the Audit Report

Use this EXACT format for the output. Always show the weighted score breakdown and always
include the TOP 3 IMPROVEMENTS section at the end.

```
============================================================
  RL GYM AUDIT REPORT
  Environment: <name>
  Audited: <datetime>
============================================================

CATEGORY 1: REWARD SIGNAL QUALITY        [X.X / 10]
------------------------------------------------------------
  [PASS] Reward range bounded 0.0-1.0
  [PASS] Dense rewards — partial credit at every step
  [WARN] Efficiency bonus present but floor too high (0.82)
  [FAIL] No grader diversity — all tasks use same grading fn
  ...
  Suggestions:
    - <specific, actionable suggestion with code reference>

CATEGORY 2: TASK COVERAGE                [X.X / 10]
------------------------------------------------------------
  ...

CATEGORY 3: OBSERVATION QUALITY          [X.X / 10]
------------------------------------------------------------
  ...

CATEGORY 4: SAFETY & ISOLATION           [X.X / 10]
------------------------------------------------------------
  ...

CATEGORY 5: API COMPLIANCE               [X.X / 10]
------------------------------------------------------------
  ...

============================================================
  SCORE BREAKDOWN
============================================================
  Category 1 — Reward Signal Quality  (30%):  X.X / 10  ->  X.XX
  Category 2 — Task Coverage          (25%):  X.X / 10  ->  X.XX
  Category 3 — Observation Quality    (20%):  X.X / 10  ->  X.XX
  Category 4 — Safety & Isolation     (15%):  X.X / 10  ->  X.XX
  Category 5 — API Compliance         (10%):  X.X / 10  ->  X.XX

============================================================
  OVERALL RL GYM QUALITY SCORE: X.X / 10   Grade: X
============================================================

TOP 3 IMPROVEMENTS (highest impact first):
  1. [CATEGORY] Specific actionable improvement
  2. [CATEGORY] Specific actionable improvement
  3. [CATEGORY] Specific actionable improvement
============================================================
```

## Important Notes

- Be honest. A score of 6/10 is not bad feedback — it's useful feedback.
- Read the actual code, don't guess. If something is ambiguous, note it as WARN with "could not verify".
- Focus on what matters for RL training quality, not code style.
- Every PASS/WARN/FAIL must have a 1-line explanation after the label.
- Suggestions must be specific: not "improve reward function" but "In `_grade_count()`, add a partial score of 0.4 when the result is within 15% of expected."
- The spec file (openenv.yaml) listing ALL tasks is a FAIL, not a WARN, if tasks are missing from the list but implemented in code.
- Read-only enforcement using `.split()` instead of regex is a WARN because it's bypassable.
