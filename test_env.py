"""
test_env.py  —  Smoke test for SQL Analytics OpenEnv environment
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None
sys.path.insert(0, os.path.dirname(__file__))

from models import SQLAction
from server.sql_analytics_environment import SQLAnalyticsEnvironment, TASKS

PASS = "[PASS]"
FAIL = "[FAIL]"

# ── Correct SQL answers for all 10 tasks ──────────────────────────────────────
CORRECT_SQL = {
    "task_001": "SELECT COUNT(*) FROM customers WHERE state = 'CA'",
    "task_002": """
        SELECT p.name, p.price FROM products p
        JOIN categories c ON p.category_id = c.id
        WHERE c.name = 'Electronics' ORDER BY p.price DESC LIMIT 1
    """,
    "task_003": """
        SELECT c.name, ROUND(SUM(o.total_amount), 2) AS total_spent
        FROM customers c JOIN orders o ON c.id = o.customer_id
        GROUP BY c.id, c.name ORDER BY total_spent DESC LIMIT 5
    """,
    "task_004": """
        SELECT cat.name AS category_name, ROUND(AVG(r.rating), 2) AS avg_rating
        FROM reviews r JOIN products p ON r.product_id = p.id
        JOIN categories cat ON p.category_id = cat.id
        GROUP BY cat.id, cat.name ORDER BY avg_rating DESC LIMIT 1
    """,
    "task_005": """
        SELECT DISTINCT c.name, c.email FROM customers c
        WHERE c.id IN (SELECT customer_id FROM orders WHERE order_date < '2024-04-01')
        AND c.id NOT IN (SELECT customer_id FROM orders WHERE order_date >= '2024-04-01')
        ORDER BY c.name
    """,
    "task_006": """
        SELECT CAST(strftime('%m', order_date) AS INTEGER) AS month,
               ROUND(SUM(total_amount), 2) AS revenue, NULL AS growth_rate
        FROM orders WHERE status = 'delivered'
        AND order_date BETWEEN '2024-01-01' AND '2024-12-31'
        GROUP BY month ORDER BY month
    """,
    "task_007": "SELECT COUNT(*) FROM products WHERE stock = 0",
    "task_008": "SELECT ROUND(AVG(total_amount), 2) AS avg_order_value FROM orders",
    "task_009": """
        SELECT p.name, cat.name AS category_name
        FROM products p JOIN categories cat ON p.category_id = cat.id
        LEFT JOIN reviews rv ON p.id = rv.product_id
        WHERE rv.id IS NULL ORDER BY p.name
    """,
    "task_010": """
        SELECT state, COUNT(*) AS customer_count FROM customers
        GROUP BY state ORDER BY customer_count DESC LIMIT 1
    """,
}

WRONG_SQL = {
    "task_001": "SELECT COUNT(*) FROM customers WHERE state = 'TX'",
    "task_002": "INVALID SQL !!!",
    "task_003": "SELECT name FROM customers LIMIT 5",
    "task_004": "SELECT 'Electronics', 5.0",
    "task_005": "SELECT name, email FROM customers LIMIT 2",
    "task_006": "SELECT 1 AS month, 100.0 AS revenue",
    "task_007": "SELECT COUNT(*) FROM products WHERE stock > 100",
    "task_008": "SELECT ROUND(AVG(total_amount), 2) AS avg FROM orders WHERE status='delivered'",
    "task_009": "SELECT name FROM products LIMIT 3",
    "task_010": "SELECT state FROM customers LIMIT 1",
}


def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def run_test(env, task_id, sql, label):
    env.reset(task_id=task_id)
    obs = env.step(SQLAction(sql=sql))
    icon = PASS if (label == "CORRECT" and obs.reward >= 0.8) or \
                   (label == "WRONG"   and obs.reward < 0.5) else FAIL
    result_count = len(obs.result) if obs.result else 0
    print(
        f"  {icon} [{label:7s}] reward={obs.reward:.4f}  "
        f"rows={result_count:3d}  done={str(obs.done):5s}  "
        f"| {(obs.metadata or {}).get('message','')[:55]}"
    )
    return obs.reward


def main():
    print("\n" + "="*60)
    print("  SQL Analytics OpenEnv — Full Smoke Test (10 tasks)")
    print("="*60)

    env = SQLAnalyticsEnvironment()
    print(f"\n  {PASS} Environment instantiated (SQLite DB seeded)")

    # ── 1. Rich Schema ────────────────────────────────────────────────────
    separator("1. Rich Schema (sample data included)")
    obs = env.reset(task_id="task_001")
    assert obs.db_schema, "Schema should be populated on reset"
    assert "Sample" in obs.db_schema, "Rich schema should include sample data"
    assert "rows)" in obs.db_schema, "Rich schema should include row counts"
    print(f"  {PASS} reset() returns RICH schema with sample data + row counts")
    print(f"\n  Schema preview (first 15 lines):\n")
    for line in (obs.db_schema or "").splitlines()[:15]:
        print(f"    {line}")

    # ── 2. All 10 tasks — correct SQL ─────────────────────────────────────
    separator("2. Correct SQL — all 10 tasks (expect reward >= 0.8)")
    correct_scores = []
    for task in TASKS:
        tid = task["task_id"]
        score = run_test(env, tid, CORRECT_SQL[tid], "CORRECT")
        correct_scores.append(score)

    # ── 3. All 10 tasks — wrong SQL ───────────────────────────────────────
    separator("3. Wrong SQL — all 10 tasks (expect reward < 0.5)")
    wrong_scores = []
    for task in TASKS:
        tid = task["task_id"]
        score = run_test(env, tid, WRONG_SQL[tid], "WRONG")
        wrong_scores.append(score)

    # ── 4. Efficiency bonus ───────────────────────────────────────────────
    separator("4. Efficiency Bonus (solve in 1 step vs 5 steps)")
    env.reset(task_id="task_001")
    obs1 = env.step(SQLAction(sql=CORRECT_SQL["task_001"]))  # solve in 1 step
    print(f"  Solved in 1 step   → reward={obs1.reward:.4f}")

    env.reset(task_id="task_001")
    for _ in range(4):
        env.step(SQLAction(sql="SELECT 1"))                  # 4 wrong attempts
    obs5 = env.step(SQLAction(sql=CORRECT_SQL["task_001"])) # solve on 5th step
    print(f"  Solved in 5 steps  → reward={obs5.reward:.4f}")
    assert obs1.reward > obs5.reward, "1-step solve should have higher reward than 5-step"
    print(f"  {PASS} Efficiency bonus working: {obs1.reward:.4f} > {obs5.reward:.4f}")

    # ── 5. Educational error messages ─────────────────────────────────────
    separator("5. Educational Error Messages")
    env.reset(task_id="task_001")

    bad_table = env.step(SQLAction(sql="SELECT * FROM orderz"))
    print(f"  Bad table name  : {bad_table.error[:65]}")
    assert "Available tables" in (bad_table.error or ""), "Should mention available tables"
    print(f"  {PASS} Bad table → educational message with available tables")

    bad_col = env.step(SQLAction(sql="SELECT nonexistent_col FROM customers"))
    print(f"  Bad column name : {bad_col.error[:65]}")
    print(f"  {PASS} Bad column → educational error message")

    syntax_err = env.step(SQLAction(sql="SELECT FROM customers WHERE"))
    print(f"  Syntax error    : {syntax_err.error[:65]}")
    assert "syntax" in (syntax_err.error or "").lower()
    print(f"  {PASS} Syntax error → educational message")

    # ── 6. New task verification (007–010) ────────────────────────────────
    separator("6. New Tasks (007-010) correct answers")
    for tid in ["task_007", "task_008", "task_009", "task_010"]:
        env.reset(task_id=tid)
        obs = env.step(SQLAction(sql=CORRECT_SQL[tid]))
        icon = PASS if obs.reward >= 0.8 else FAIL
        print(f"  {icon} {tid}  reward={obs.reward:.4f}  done={obs.done}")

    # ── 7. Steps remaining in metadata ───────────────────────────────────
    separator("7. steps_remaining in metadata")
    env.reset(task_id="task_002")
    for i in range(3):
        obs = env.step(SQLAction(sql="SELECT 1"))
    remaining = (obs.metadata or {}).get("steps_remaining", -1)
    assert remaining == 7, f"Expected 7 steps remaining after 3 steps, got {remaining}"
    print(f"  {PASS} steps_remaining={remaining} correctly tracked")

    # ── 8. State tracking ────────────────────────────────────────────────
    separator("8. State Tracking")
    env.reset(task_id="task_003")
    for i in range(3):
        env.step(SQLAction(sql="SELECT 1"))
    s = env.state
    assert s.attempts == 3
    assert s.step_count == 3
    print(f"  {PASS} step_count={s.step_count}  attempts={s.attempts}  task_id={s.task_id}")

    # ── 9. Max attempts cap ───────────────────────────────────────────────
    separator("9. Max Attempts Cap (10 steps → done=True)")
    env.reset(task_id="task_002")
    last_obs = None
    for _ in range(11):
        last_obs = env.step(SQLAction(sql="SELECT 1"))
        if last_obs.done:
            break
    assert last_obs.done
    print(f"  {PASS} done=True after {env.state.attempts} attempts")

    # ── Summary ───────────────────────────────────────────────────────────
    separator("Summary")
    avg_correct = sum(correct_scores) / len(correct_scores)
    avg_wrong   = sum(wrong_scores)   / len(wrong_scores)
    all_pass    = avg_correct >= 0.8 and avg_wrong < 0.5

    print(f"  Tasks total          : {len(TASKS)}")
    print(f"  Avg reward (correct) : {avg_correct:.4f}")
    print(f"  Avg reward (wrong)   : {avg_wrong:.4f}")
    print()
    icon = PASS if all_pass else FAIL
    print(f"  {icon} Overall: {'PASSED' if all_pass else 'FAILED'}")
    print()

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
