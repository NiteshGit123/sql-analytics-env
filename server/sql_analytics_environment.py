"""
sql_analytics_environment.py
-----------------------------
Core environment implementation for the SQL Analytics OpenEnv environment.

An AI agent answers real-world business-intelligence questions by writing
SQL queries against a seeded e-commerce SQLite database.

Database schema
---------------
  customers(id, name, email, state, signup_date, last_order_date)
  categories(id, name)
  products(id, name, category_id, price, stock)
  orders(id, customer_id, order_date, total_amount, status)
  order_items(id, order_id, product_id, quantity, unit_price)
  reviews(id, product_id, customer_id, rating, review_date)

Six tasks (easy x2 / medium x2 / hard x2) with deterministic expected
answers.  Rewards carry partial-progress signals so an RL agent can learn
from near-misses.
"""

from __future__ import annotations

import logging
import random
import re
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

try:
    from openenv.core.env_server.interfaces import Environment
    from ..models import SQLAction, SQLObservation, SQLState
except ImportError:
    from openenv.core.env_server.interfaces import Environment
    from models import SQLAction, SQLObservation, SQLState

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger("sql_analytics_env")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# ── Query safety ──────────────────────────────────────────────────────────────
_QUERY_TIMEOUT_SECONDS = 5          # max seconds per SQL query
_BLOCKED_KEYWORDS = frozenset([     # read-only enforcement
    "insert", "update", "delete", "drop", "create", "alter",
    "attach", "detach", "pragma", "vacuum",
])


# ── Schema ─────────────────────────────────────────────────────────────────────

_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS customers (
    id             INTEGER PRIMARY KEY,
    name           TEXT    NOT NULL,
    email          TEXT    NOT NULL,
    state          TEXT    NOT NULL,
    signup_date    TEXT    NOT NULL,
    last_order_date TEXT
);

CREATE TABLE IF NOT EXISTS categories (
    id   INTEGER PRIMARY KEY,
    name TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS products (
    id          INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL,
    category_id INTEGER NOT NULL REFERENCES categories(id),
    price       REAL    NOT NULL,
    stock       INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    id           INTEGER PRIMARY KEY,
    customer_id  INTEGER NOT NULL REFERENCES customers(id),
    order_date   TEXT    NOT NULL,
    total_amount REAL    NOT NULL,
    status       TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS order_items (
    id         INTEGER PRIMARY KEY,
    order_id   INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity   INTEGER NOT NULL,
    unit_price REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS reviews (
    id          INTEGER PRIMARY KEY,
    product_id  INTEGER NOT NULL REFERENCES products(id),
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    rating      INTEGER NOT NULL,
    review_date TEXT    NOT NULL
);
"""

# Static schema description — runtime-enriched version built after DB seed
_SCHEMA_DESCRIPTION = """
Database Schema (SQLite, read-only):
=======================================================

customers(id, name, email, state, signup_date, last_order_date)
  - state          : 2-letter US state code  (e.g. 'CA', 'NY', 'TX')
  - signup_date    : 'YYYY-MM-DD'
  - last_order_date: 'YYYY-MM-DD' or NULL

categories(id, name)
  - name values: 'Electronics', 'Clothing', 'Books', 'Food', 'Sports'

products(id, name, category_id, price, stock)
  - category_id references categories.id

orders(id, customer_id, order_date, total_amount, status)
  - order_date  : 'YYYY-MM-DD'
  - status      : 'delivered' | 'shipped' | 'pending' | 'cancelled'
  - customer_id references customers.id

order_items(id, order_id, product_id, quantity, unit_price)
  - order_id   references orders.id
  - product_id references products.id

reviews(id, product_id, customer_id, rating, review_date)
  - rating     : integer 1-5
  - review_date: 'YYYY-MM-DD'
  - product_id references products.id
  - customer_id references customers.id
""".strip()


def _build_rich_schema(conn: sqlite3.Connection) -> str:
    """
    Build a rich schema description with row counts and sample data.
    This is much more useful for an AI agent than just column names —
    it can see what actual values look like (e.g. 'CA' not 'California').
    """
    conn.row_factory = sqlite3.Row

    tables = [
        ("customers",    "id, name, email, state, signup_date, last_order_date"),
        ("categories",   "id, name"),
        ("products",     "id, name, category_id, price, stock"),
        ("orders",       "id, customer_id, order_date, total_amount, status"),
        ("order_items",  "id, order_id, product_id, quantity, unit_price"),
        ("reviews",      "id, product_id, customer_id, rating, review_date"),
    ]

    notes = {
        "customers":   "state is 2-letter US code. signup_date/last_order_date are 'YYYY-MM-DD'.",
        "categories":  "name is one of: 'Electronics', 'Clothing', 'Books', 'Food', 'Sports'.",
        "products":    "category_id references categories.id. price is REAL.",
        "orders":      "status is one of: 'delivered', 'shipped', 'pending', 'cancelled'.",
        "order_items": "Links orders to products. unit_price is the price at time of order.",
        "reviews":     "rating is integer 1-5. review_date is 'YYYY-MM-DD'.",
    }

    lines = [
        "Database Schema (SQLite, read-only):",
        "=" * 55,
        "",
    ]

    for table, cols in tables:
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        lines.append(f"TABLE: {table}  ({row_count:,} rows)")
        lines.append(f"  Columns : {cols}")
        lines.append(f"  Notes   : {notes[table]}")

        # Sample rows (first 3)
        sample_rows = conn.execute(
            f"SELECT {cols} FROM {table} LIMIT 3"
        ).fetchall()
        if sample_rows:
            lines.append("  Sample  :")
            col_names = cols.split(", ")
            for r in sample_rows:
                row_str = " | ".join(f"{col}={str(r[col])[:18]}" for col in col_names)
                lines.append(f"    {row_str}")
        lines.append("")

    return "\n".join(lines).strip()


def _educational_error(sql: str, raw_error: str) -> str:
    """
    Convert a raw SQLite error into a human-readable educational message
    that helps the agent understand what went wrong and how to fix it.
    """
    e = raw_error.lower()
    available_tables = "customers, categories, products, orders, order_items, reviews"

    if "no such table" in e:
        bad = raw_error.split(":")[-1].strip()
        return (
            f"Table '{bad}' does not exist. "
            f"Available tables: {available_tables}. "
            "Check your spelling and use lowercase table names."
        )
    if "no such column" in e:
        return (
            f"{raw_error}. "
            "Check the schema for the correct column name. "
            "Tip: use table_name.column_name to avoid ambiguity."
        )
    if "ambiguous column" in e:
        bad = raw_error.split(":")[-1].strip()
        return (
            f"Column '{bad}' is ambiguous — it exists in multiple tables. "
            "Qualify it with the table name, e.g. customers.id or orders.id."
        )
    if "syntax error" in e or "near" in e:
        return (
            f"SQL syntax error: {raw_error}. "
            "Common causes: missing comma, unmatched parenthesis, "
            "misspelled keyword, or missing FROM clause."
        )
    if "not unique" in e or "unique constraint" in e:
        return f"Duplicate value error: {raw_error}. You may need DISTINCT."
    if "no such function" in e:
        fn = raw_error.split(":")[-1].strip()
        return (
            f"Function '{fn}' is not available in SQLite. "
            "SQLite supports: COUNT, SUM, AVG, MIN, MAX, ROUND, "
            "strftime, COALESCE, IFNULL, CAST."
        )

    return f"SQL error: {raw_error}"


# ── Database seeding ───────────────────────────────────────────────────────────

def _seed(conn: sqlite3.Connection) -> None:
    """Populate the database with deterministic synthetic e-commerce data."""
    rng = random.Random(42)

    # ── categories ──
    categories = ["Electronics", "Clothing", "Books", "Food", "Sports"]
    conn.executemany(
        "INSERT INTO categories (id, name) VALUES (?, ?)",
        [(i + 1, n) for i, n in enumerate(categories)],
    )

    # ── products (10 per category = 50 total) ──
    _PRODUCTS: Dict[int, List[Tuple[str, float]]] = {
        1: [  # Electronics
            ("iPhone 15", 999.99), ("Samsung 55\" TV", 799.99),
            ("MacBook Pro 14\"", 2499.99), ("AirPods Pro", 249.99),
            ("iPad Air", 599.99), ("Sony WH-1000XM5", 349.99),
            ("Dell 27\" Monitor", 399.99), ("Mechanical Keyboard", 149.99),
            ("Logitech MX Master", 99.99), ("USB-C Hub 7-port", 49.99),
        ],
        2: [  # Clothing
            ("Running Shoes", 129.99), ("Slim Fit Jeans", 59.99),
            ("Winter Parka", 199.99), ("Yoga Pants", 49.99),
            ("Classic T-Shirt", 24.99), ("Formal Blazer", 299.99),
            ("Canvas Sneakers", 89.99), ("Floral Dress", 79.99),
            ("Zip Hoodie", 54.99), ("Board Shorts", 34.99),
        ],
        3: [  # Books
            ("Python Cookbook 3e", 39.99), ("Clean Code", 44.99),
            ("Deep Learning (Goodfellow)", 54.99), ("The Pragmatic Programmer", 49.99),
            ("Design Patterns (GoF)", 44.99), ("Atomic Habits", 19.99),
            ("Zero to One", 24.99), ("Sapiens", 22.99),
            ("The Art of War", 12.99), ("Thinking Fast and Slow", 17.99),
        ],
        4: [  # Food
            ("Organic Coffee 1 kg", 24.99), ("Matcha Green Tea Set", 19.99),
            ("Whey Protein 2 kg", 49.99), ("Granola Bar 24-pack", 14.99),
            ("Extra Virgin Olive Oil", 18.99), ("70% Dark Chocolate Box", 22.99),
            ("Roasted Almonds 500 g", 16.99), ("Artisan Pasta 500 g", 8.99),
            ("Craft Hot Sauce Set", 29.99), ("Raw Organic Honey", 12.99),
        ],
        5: [  # Sports
            ("Yoga Mat 6mm", 49.99), ("Resistance Band Set", 29.99),
            ("Adjustable Dumbbell 20 kg", 89.99), ("Tennis Racket Pro", 79.99),
            ("Official Basketball", 34.99), ("Speed Jump Rope", 14.99),
            ("Foam Roller 60cm", 24.99), ("Doorframe Pull-up Bar", 39.99),
            ("Padded Gym Gloves", 19.99), ("40 oz Insulated Bottle", 29.99),
        ],
    }

    products: List[Tuple] = []
    pid = 1
    for cat_id, items in _PRODUCTS.items():
        for name, price in items:
            products.append((pid, name, cat_id, price, rng.randint(0, 200)))
            pid += 1
    conn.executemany(
        "INSERT INTO products (id, name, category_id, price, stock) VALUES (?,?,?,?,?)",
        products,
    )

    # ── customers (100) ──
    _FIRST = ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace",
              "Henry", "Iris", "Jack", "Karen", "Leo", "Mary", "Nick",
              "Olivia", "Paul", "Quinn", "Rachel", "Sam", "Tina"]
    _LAST  = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
              "Miller", "Davis", "Martinez", "Wilson", "Anderson", "Taylor",
              "Thomas", "Moore", "Jackson", "Martin", "Lee", "Perez",
              "Thompson", "White"]
    _STATES = (["CA"] * 20 + ["NY"] * 12 + ["TX"] * 12 + ["FL"] * 8 +
               ["WA"] * 6 + ["IL"] * 5 + ["GA"] * 5 + ["MA"] * 5 +
               ["CO"] * 4 + ["AZ"] * 4 + ["PA"] * 4 + ["OR"] * 3 +
               ["NV"] * 3 + ["OH"] * 3 + ["MI"] * 3 + ["NC"] * 3)

    BASE = datetime(2023, 1, 1)
    customers: List[Tuple] = []
    for i in range(100):
        fn = rng.choice(_FIRST)
        ln = rng.choice(_LAST)
        st = rng.choice(_STATES)
        signup = (BASE + timedelta(days=rng.randint(0, 365))).strftime("%Y-%m-%d")
        customers.append((i + 1, f"{fn} {ln}",
                          f"{fn.lower()}.{ln.lower()}{i}@example.com",
                          st, signup, None))
    conn.executemany(
        "INSERT INTO customers (id, name, email, state, signup_date, last_order_date)"
        " VALUES (?,?,?,?,?,?)",
        customers,
    )

    # ── orders (500) and order_items ──
    orders: List[Tuple] = []
    items_rows: List[Tuple] = []
    oid = 1
    iid = 1
    last_order: Dict[int, str] = {}
    YEAR_START = datetime(2024, 1, 1)

    for _ in range(500):
        cust_id = rng.randint(1, 100)
        odate = (YEAR_START + timedelta(days=rng.randint(0, 364))).strftime("%Y-%m-%d")
        status = rng.choices(
            ["delivered", "shipped", "pending", "cancelled"],
            weights=[60, 20, 10, 10],
        )[0]

        n_items = rng.randint(1, 4)
        chosen = rng.sample(range(1, 51), n_items)
        total = 0.0
        for prod_id in chosen:
            qty = rng.randint(1, 3)
            price = products[prod_id - 1][3]
            total += qty * price
            items_rows.append((iid, oid, prod_id, qty, price))
            iid += 1

        total = round(total, 2)
        orders.append((oid, cust_id, odate, total, status))

        if cust_id not in last_order or odate > last_order[cust_id]:
            last_order[cust_id] = odate
        oid += 1

    conn.executemany(
        "INSERT INTO orders (id, customer_id, order_date, total_amount, status)"
        " VALUES (?,?,?,?,?)",
        orders,
    )
    conn.executemany(
        "INSERT INTO order_items (id, order_id, product_id, quantity, unit_price)"
        " VALUES (?,?,?,?,?)",
        items_rows,
    )

    for cid, ldate in last_order.items():
        conn.execute(
            "UPDATE customers SET last_order_date = ? WHERE id = ?", (ldate, cid)
        )

    # ── reviews (300) ──
    reviews: List[Tuple] = []
    for i in range(300):
        prod_id = rng.randint(1, 50)
        cust_id = rng.randint(1, 100)
        rating = rng.choices([1, 2, 3, 4, 5], weights=[5, 10, 20, 35, 30])[0]
        rdate = (YEAR_START + timedelta(days=rng.randint(0, 364))).strftime("%Y-%m-%d")
        reviews.append((i + 1, prod_id, cust_id, rating, rdate))
    conn.executemany(
        "INSERT INTO reviews (id, product_id, customer_id, rating, review_date)"
        " VALUES (?,?,?,?,?)",
        reviews,
    )
    conn.commit()


# ── Task definitions ───────────────────────────────────────────────────────────

TASKS = [
    # ── EASY ──────────────────────────────────────────────────────────────────
    {
        "task_id":    "task_001",
        "difficulty": "easy",
        "description": (
            "Count the total number of customers registered in the state of "
            "California (state = 'CA'). Return a single integer value."
        ),
        "hint": "Use the customers table and filter WHERE state = 'CA'.",
    },
    {
        "task_id":    "task_002",
        "difficulty": "easy",
        "description": (
            "Find the name and price of the most expensive product in the "
            "'Electronics' category. Return exactly one row with columns: "
            "name, price."
        ),
        "hint": (
            "JOIN products with categories, filter WHERE categories.name = "
            "'Electronics', ORDER BY price DESC, LIMIT 1."
        ),
    },
    # ── MEDIUM ────────────────────────────────────────────────────────────────
    {
        "task_id":    "task_003",
        "difficulty": "medium",
        "description": (
            "List the top 5 customers by total amount spent across all their "
            "orders (any status). Return columns: name, total_spent "
            "(rounded to 2 decimal places). Order by total_spent descending."
        ),
        "hint": (
            "JOIN customers with orders. GROUP BY customer. "
            "SUM(total_amount). ORDER BY total_spent DESC. LIMIT 5."
        ),
    },
    {
        "task_id":    "task_004",
        "difficulty": "medium",
        "description": (
            "Which product category has the highest average customer rating? "
            "Return columns: category_name, avg_rating "
            "(rounded to 2 decimal places). Return only the top category."
        ),
        "hint": (
            "JOIN reviews → products → categories. "
            "GROUP BY category. AVG(rating). ORDER BY avg_rating DESC. LIMIT 1."
        ),
    },
    # ── HARD ──────────────────────────────────────────────────────────────────
    {
        "task_id":    "task_005",
        "difficulty": "hard",
        "description": (
            "Find all customers who placed at least one order in Q1 2024 "
            "(January-March, i.e. order_date < '2024-04-01') but have NOT "
            "placed any order from April 2024 onwards "
            "(order_date >= '2024-04-01'). "
            "Return columns: name, email. Order alphabetically by name."
        ),
        "hint": (
            "Use subqueries or EXCEPT. One subquery selects customer_ids with "
            "orders before '2024-04-01'; another selects customer_ids with "
            "orders from '2024-04-01' onward. Take the difference."
        ),
    },
    {
        "task_id":    "task_006",
        "difficulty": "hard",
        "description": (
            "Calculate monthly revenue (sum of total_amount) for each month "
            "in 2024 using only 'delivered' orders. Then compute the "
            "month-over-month growth rate as a percentage. "
            "Return columns: month (integer 1-12), revenue "
            "(rounded to 2 decimal places), growth_rate (percentage change "
            "vs previous month rounded to 2 decimal places; NULL for the "
            "first month). Order by month ascending. "
            "Only include months that have at least one delivered order."
        ),
        "hint": (
            "Use strftime('%m', order_date) to extract month. "
            "Filter status = 'delivered'. For growth_rate use a self-join or "
            "window function LAG() if SQLite version supports it."
        ),
    },

    # ── EASY (additional) ─────────────────────────────────────────────────────
    {
        "task_id":    "task_007",
        "difficulty": "easy",
        "description": (
            "How many products are currently out of stock (stock = 0)? "
            "Return a single integer value."
        ),
        "hint": "Use the products table. Filter WHERE stock = 0. COUNT(*).",
    },
    {
        "task_id":    "task_008",
        "difficulty": "easy",
        "description": (
            "What is the average order value (total_amount) across ALL orders "
            "regardless of status? Return a single value rounded to 2 decimal "
            "places. Column name: avg_order_value."
        ),
        "hint": "Use the orders table. AVG(total_amount). ROUND to 2 decimal places.",
    },

    # ── MEDIUM (additional) ───────────────────────────────────────────────────
    {
        "task_id":    "task_009",
        "difficulty": "medium",
        "description": (
            "List all products that have NEVER received any customer review. "
            "Return columns: name, category_name. "
            "Order by product name alphabetically."
        ),
        "hint": (
            "LEFT JOIN products with reviews ON products.id = reviews.product_id. "
            "Filter WHERE reviews.id IS NULL (no matching review). "
            "JOIN categories to get category_name."
        ),
    },
    {
        "task_id":    "task_010",
        "difficulty": "medium",
        "description": (
            "Which US state has the highest number of registered customers? "
            "Return columns: state, customer_count. Return only the top 1 state."
        ),
        "hint": (
            "Use the customers table. GROUP BY state. COUNT(*) AS customer_count. "
            "ORDER BY customer_count DESC. LIMIT 1."
        ),
    },
]


# ── Grading helpers ────────────────────────────────────────────────────────────

def _rows_to_dicts(rows: List[sqlite3.Row]) -> List[Dict[str, Any]]:
    return [dict(r) for r in rows]


def _lower_keys(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k.lower(): v for k, v in row.items()}


def _find_col(row: Dict[str, Any], *hints: str) -> Optional[Any]:
    """Return the value of the first column whose name contains any hint."""
    for hint in hints:
        for k, v in row.items():
            if hint in k:
                return v
    return None


def _grade_count(result: List, expected: int) -> Tuple[float, bool, str]:
    if not result:
        return 0.1, False, "Empty result; expected a single count row."
    row = _lower_keys(result[0])
    raw = next(iter(row.values()), None)
    try:
        got = int(raw)
    except (TypeError, ValueError):
        return 0.2, False, f"Cannot parse integer from first column: {raw!r}"

    if got == expected:
        return 1.0, True, f"Correct! Count = {got}."
    diff = abs(got - expected) / max(expected, 1)
    if diff <= 0.02:
        return 0.8, False, f"Very close: got {got}, expected {expected}."
    elif diff <= 0.15:
        return 0.4, False, f"Somewhat close: got {got}, expected {expected}."
    return 0.1, False, f"Incorrect: got {got}, expected {expected}."


def _grade_single_row(
    result: List, expected: Dict[str, Any]
) -> Tuple[float, bool, str]:
    if not result:
        return 0.1, False, "Empty result; expected one row."
    row = _lower_keys(result[0])
    hits = 0
    for key, exp_val in expected.items():
        # Flexible column matching
        got_val = row.get(key.lower())
        if got_val is None:
            for k in row:
                if key.lower() in k or k in key.lower():
                    got_val = row[k]
                    break
        if got_val is None:
            continue
        if isinstance(exp_val, float):
            try:
                if abs(float(got_val) - exp_val) / max(abs(exp_val), 0.01) < 0.01:
                    hits += 1
            except (TypeError, ValueError):
                pass
        else:
            if str(got_val).strip().lower() == str(exp_val).strip().lower():
                hits += 1
    score = hits / len(expected)
    return score, score >= 1.0, f"Matched {hits}/{len(expected)} expected fields."


def _grade_top_n(
    result: List,
    expected: List[Dict],
    name_col: str,
    value_col: str,
) -> Tuple[float, bool, str]:
    if not result:
        return 0.1, False, "Empty result; expected multiple rows."
    n = len(expected)
    result_low = [_lower_keys(r) for r in result]

    got_names = []
    for row in result_low:
        v = _find_col(row, name_col, "name", "customer")
        if v is not None:
            got_names.append(str(v))

    exp_names = [str(e[name_col]) for e in expected]

    if not got_names:
        return 0.2, False, f"Could not find '{name_col}' column in result."

    # Exact order
    if got_names[:n] == exp_names:
        return 1.0, True, f"Perfect top-{n} in correct order."

    overlap = len(set(got_names[:n]) & set(exp_names))
    pos_matches = sum(
        1 for i, g in enumerate(got_names[:n])
        if i < len(exp_names) and g == exp_names[i]
    )
    score = 0.3 + 0.5 * (overlap / n) + 0.2 * (pos_matches / n)
    return min(score, 0.95), False, f"Partial: {overlap}/{n} names, {pos_matches}/{n} in correct position."


def _grade_set_match(
    result: List,
    expected: List[Dict],
    col: str = "name",
) -> Tuple[float, bool, str]:
    result_low = [_lower_keys(r) for r in result]
    got_set = set()
    for row in result_low:
        v = _find_col(row, col, "name", "customer")
        if v is not None:
            got_set.add(str(v).strip())
    exp_set = {str(e[col]).strip() for e in expected}

    if not exp_set and not got_set:
        return 1.0, True, "Correct: no rows expected and none returned."
    if not exp_set:
        return 0.0, False, f"Expected 0 rows but got {len(got_set)}."
    if not got_set:
        return 0.1, False, "Empty result but expected rows."

    precision = len(got_set & exp_set) / len(got_set)
    recall    = len(got_set & exp_set) / len(exp_set)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return f1, f1 >= 0.95, (
        f"F1={f1:.2f} (precision={precision:.2f}, recall={recall:.2f}). "
        f"Expected {len(exp_set)} rows, got {len(got_set)}."
    )


def _grade_monthly_revenue(
    result: List, expected: List[Dict]
) -> Tuple[float, bool, str]:
    if not result or len(result) < 2:
        return 0.1, False, f"Too few rows ({len(result)}); expected ~{len(expected)}."

    result_low = [_lower_keys(r) for r in result]
    got_months: Dict[int, float] = {}
    for row in result_low:
        m = _find_col(row, "month")
        r = _find_col(row, "revenue", "amount", "total", "sum")
        if m is not None and r is not None:
            try:
                got_months[int(m)] = float(r)
            except (TypeError, ValueError):
                pass

    if not got_months:
        return 0.2, False, "Could not extract month/revenue columns."

    exp_months = {e["month"]: e["revenue"] for e in expected}

    # Revenue accuracy
    rev_scores = []
    for mo, exp_rev in exp_months.items():
        got_rev = got_months.get(mo)
        if got_rev is None:
            rev_scores.append(0.0)
        else:
            diff = abs(got_rev - exp_rev) / max(abs(exp_rev), 1.0)
            rev_scores.append(max(0.0, 1.0 - diff * 5))

    rev_score = sum(rev_scores) / len(rev_scores)

    # Bonus: growth_rate column present
    has_growth = any(
        any("growth" in k or "rate" in k for k in _lower_keys(r))
        for r in result
    )
    score = 0.75 * rev_score + (0.25 if has_growth else 0.0)
    done  = score >= 0.85
    return round(score, 4), done, (
        f"Revenue accuracy={rev_score:.2f}, growth_rate column present={has_growth}."
    )


# ── Environment ────────────────────────────────────────────────────────────────

class SQLAnalyticsEnvironment(Environment):
    """
    OpenEnv environment: SQL Analytics over an e-commerce database.

    An AI agent receives a natural-language BI question and must write
    correct SQL queries against the database to find the answer.

    Interaction loop
    ----------------
    1. ``reset(task_id="task_001")``  →  initial SQLObservation with schema + task
    2. ``step(SQLAction(sql="SELECT ..."))``  →  query result + partial reward
    3. Repeat step 2 until ``observation.done == True``

    Tasks (difficulty)
    ------------------
    task_001  easy    count customers in CA
    task_002  easy    most expensive Electronics product
    task_003  medium  top 5 customers by spend
    task_004  medium  category with highest avg rating
    task_005  hard    Q1-only buyers (churn signal)
    task_006  hard    monthly revenue + growth rate
    """

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self) -> None:
        self._conn: Optional[sqlite3.Connection] = None
        self._state = SQLState(episode_id=str(uuid4()), step_count=0)
        self._current_task: Optional[Dict] = None
        self._answers: Dict[str, Any] = {}
        self._setup()

    # ── setup ──────────────────────────────────────────────────────────────

    def _setup(self) -> None:
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA_DDL)
        _seed(self._conn)
        # Build rich schema AFTER seeding so sample rows are available
        self._rich_schema = _build_rich_schema(self._conn)
        self._precompute_answers()

    def _precompute_answers(self) -> None:
        c = self._conn

        # task_001
        r = c.execute(
            "SELECT COUNT(*) AS cnt FROM customers WHERE state = 'CA'"
        ).fetchone()
        self._answers["task_001"] = int(r["cnt"])

        # task_002
        r = c.execute("""
            SELECT p.name, p.price
            FROM products p
            JOIN categories cat ON p.category_id = cat.id
            WHERE cat.name = 'Electronics'
            ORDER BY p.price DESC
            LIMIT 1
        """).fetchone()
        self._answers["task_002"] = {"name": r["name"], "price": r["price"]}

        # task_003
        rows = c.execute("""
            SELECT cu.name, ROUND(SUM(o.total_amount), 2) AS total_spent
            FROM customers cu
            JOIN orders o ON cu.id = o.customer_id
            GROUP BY cu.id, cu.name
            ORDER BY total_spent DESC
            LIMIT 5
        """).fetchall()
        self._answers["task_003"] = _rows_to_dicts(rows)

        # task_004
        r = c.execute("""
            SELECT cat.name AS category_name, ROUND(AVG(rv.rating), 2) AS avg_rating
            FROM reviews rv
            JOIN products p  ON rv.product_id  = p.id
            JOIN categories cat ON p.category_id = cat.id
            GROUP BY cat.id, cat.name
            ORDER BY avg_rating DESC
            LIMIT 1
        """).fetchone()
        self._answers["task_004"] = {
            "category_name": r["category_name"],
            "avg_rating": r["avg_rating"],
        }

        # task_005
        rows = c.execute("""
            SELECT DISTINCT cu.name, cu.email
            FROM customers cu
            WHERE cu.id IN (
                SELECT customer_id FROM orders
                WHERE order_date >= '2024-01-01' AND order_date < '2024-04-01'
            )
            AND cu.id NOT IN (
                SELECT customer_id FROM orders
                WHERE order_date >= '2024-04-01'
            )
            ORDER BY cu.name
        """).fetchall()
        self._answers["task_005"] = _rows_to_dicts(rows)

        # task_006
        rows = c.execute("""
            SELECT CAST(strftime('%m', order_date) AS INTEGER) AS month,
                   ROUND(SUM(total_amount), 2) AS revenue
            FROM orders
            WHERE status = 'delivered'
              AND order_date BETWEEN '2024-01-01' AND '2024-12-31'
            GROUP BY month
            ORDER BY month
        """).fetchall()
        monthly: List[Dict] = _rows_to_dicts(rows)
        for i, m in enumerate(monthly):
            if i == 0:
                m["growth_rate"] = None
            else:
                prev = monthly[i - 1]["revenue"]
                m["growth_rate"] = (
                    round((m["revenue"] - prev) / prev * 100, 2)
                    if prev else None
                )
        self._answers["task_006"] = monthly

        # task_007 — out of stock count
        r = c.execute("SELECT COUNT(*) AS cnt FROM products WHERE stock = 0").fetchone()
        self._answers["task_007"] = int(r["cnt"])

        # task_008 — average order value
        r = c.execute(
            "SELECT ROUND(AVG(total_amount), 2) AS avg_order_value FROM orders"
        ).fetchone()
        self._answers["task_008"] = {"avg_order_value": r["avg_order_value"]}

        # task_009 — products never reviewed
        rows = c.execute("""
            SELECT p.name, cat.name AS category_name
            FROM products p
            JOIN categories cat ON p.category_id = cat.id
            LEFT JOIN reviews rv ON p.id = rv.product_id
            WHERE rv.id IS NULL
            ORDER BY p.name
        """).fetchall()
        self._answers["task_009"] = _rows_to_dicts(rows)

        # task_010 — state with most customers
        r = c.execute("""
            SELECT state, COUNT(*) AS customer_count
            FROM customers
            GROUP BY state
            ORDER BY customer_count DESC
            LIMIT 1
        """).fetchone()
        self._answers["task_010"] = {"state": r["state"], "customer_count": r["customer_count"]}

    # ── OpenEnv API ────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SQLObservation:
        """Start a new episode for the given task_id (default: task_001)."""
        task_id = task_id or "task_001"
        task_def = next(
            (t for t in TASKS if t["task_id"] == task_id), TASKS[0]
        )

        self._current_task = task_def
        self._state = SQLState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task_def["task_id"],
            task_description=task_def["description"],
            difficulty=task_def["difficulty"],
            attempts=0,
            max_attempts=10,
            solved=False,
        )

        return SQLObservation(
            done=False,
            reward=0.0,
            result=None,
            error=None,
            db_schema=self._rich_schema,   # ← rich schema with samples
            task_description=task_def["description"],
            metadata={
                "task_id":    task_def["task_id"],
                "difficulty": task_def["difficulty"],
                "hint":       task_def["hint"],
                "max_steps":  10,
                "message": (
                    "Environment ready. Submit a SQL query via SQLAction(sql=...) "
                    "to answer the task. Tip: solving in fewer steps earns an "
                    "efficiency bonus on top of your base reward."
                ),
            },
        )

    def step(
        self,
        action: SQLAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SQLObservation:
        """Execute the agent's SQL query and return a graded observation."""
        if self._current_task is None:
            return SQLObservation(
                done=True,
                reward=0.0,
                error="No task loaded. Call reset(task_id=...) first.",
                metadata={"message": "Call reset() before step()."},
            )

        self._state.step_count += 1
        self._state.attempts += 1

        # Execute SQL (read-only, with safety checks + timeout)
        sql = (action.sql or "").strip()
        result: Optional[List] = None
        error: Optional[str] = None

        # Block write operations
        sql_lower = sql.lower()
        blocked = [kw for kw in _BLOCKED_KEYWORDS if re.search(r'\b' + kw + r'\b', sql_lower)]
        if blocked:
            error = f"Write operations are not allowed: {blocked}. Only SELECT queries are permitted."
            logger.warning("Blocked SQL keyword(s) %s in episode=%s", blocked, self._state.episode_id)
        elif not sql:
            error = "Empty SQL query submitted."
        else:
            # Run query in a thread with a timeout
            _result_box: List = []
            _error_box:  List = []

            def _run_query():
                try:
                    cur = self._conn.execute(sql)
                    rows = cur.fetchall()
                    _result_box.append(_rows_to_dicts(rows))
                except Exception as exc:
                    _error_box.append(str(exc))

            t = threading.Thread(target=_run_query, daemon=True)
            t.start()
            t.join(timeout=_QUERY_TIMEOUT_SECONDS)

            if t.is_alive():
                error = f"Query timed out after {_QUERY_TIMEOUT_SECONDS}s."
                logger.warning("Query timeout in episode=%s | sql=%s", self._state.episode_id, sql[:80])
            elif _error_box:
                # Convert raw error → educational message
                error = _educational_error(sql, _error_box[0])
            else:
                result = _result_box[0] if _result_box else []

        logger.info(
            "step | episode=%s | task=%s | attempt=%d | rows=%s | reward_pending | error=%s",
            self._state.episode_id,
            self._state.task_id,
            self._state.attempts,
            len(result) if result is not None else "N/A",
            error,
        )

        # Grade
        reward, done, message = self._grade(result, error)

        # ── Efficiency bonus ────────────────────────────────────────────────
        # Reward agents that solve correctly in fewer steps.
        # Solving in 1 step → reward stays 1.0 (no penalty)
        # Solving in 5 steps → base * 0.92
        # Solving in 10 steps → base * 0.82
        # Only applied when the task is actually solved (base reward = 1.0)
        if done and reward >= 0.9:
            efficiency = max(0.82, 1.0 - (self._state.attempts - 1) * 0.02)
            reward = round(reward * efficiency, 4)
            if self._state.attempts == 1:
                message += " [+efficiency bonus: solved in 1 step!]"
            elif efficiency < 1.0:
                message += f" [efficiency={efficiency:.2f}: solved in {self._state.attempts} steps]"

        if done:
            self._state.solved = reward >= 0.82   # still counts as solved after bonus

        # Enforce max-attempts cap
        if self._state.attempts >= self._state.max_attempts and not done:
            done = True
            message += f" [Max attempts ({self._state.max_attempts}) reached.]"

        return SQLObservation(
            done=done,
            reward=reward,
            result=result,
            error=error,
            db_schema=self._rich_schema,
            task_description=self._current_task["description"],
            metadata={
                "task_id":    self._current_task["task_id"],
                "difficulty": self._current_task["difficulty"],
                "attempts":   self._state.attempts,
                "max_attempts": self._state.max_attempts,
                "steps_remaining": max(0, self._state.max_attempts - self._state.attempts),
                "message":    message,
            },
        )

    @property
    def state(self) -> SQLState:
        return self._state

    # ── Grading dispatch ───────────────────────────────────────────────────

    def _grade(
        self,
        result: Optional[List],
        error: Optional[str],
    ) -> Tuple[float, bool, str]:
        task_id = self._current_task["task_id"]

        if error:
            # Return 0.0 but keep done=False so agent can retry
            return 0.0, False, error
        if result is None:
            return 0.0, False, "No result returned."

        if task_id == "task_001":
            return _grade_count(result, self._answers["task_001"])
        elif task_id == "task_002":
            return _grade_single_row(result, self._answers["task_002"])
        elif task_id == "task_003":
            return _grade_top_n(
                result, self._answers["task_003"],
                name_col="name", value_col="total_spent",
            )
        elif task_id == "task_004":
            return _grade_single_row(result, self._answers["task_004"])
        elif task_id == "task_005":
            return _grade_set_match(
                result, self._answers["task_005"], col="name"
            )
        elif task_id == "task_006":
            return _grade_monthly_revenue(result, self._answers["task_006"])
        elif task_id == "task_007":
            return _grade_count(result, self._answers["task_007"])
        elif task_id == "task_008":
            return _grade_single_row(result, self._answers["task_008"])
        elif task_id == "task_009":
            return _grade_set_match(
                result, self._answers["task_009"], col="name"
            )
        elif task_id == "task_010":
            return _grade_single_row(result, self._answers["task_010"])

        return 0.0, False, f"Unknown task_id: {task_id}"
