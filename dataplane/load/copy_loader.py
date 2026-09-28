"""
dataplane/load/copy_loader.py
=============================

WHAT THIS FILE IS FOR
---------------------
Takes the generated tables (polars DataFrames held in memory) and loads
them into the mf_data database.

WHY COPY, AND NOT INSERT
------------------------
INSERT sends rows one statement at a time. At 800,000 transactions that is
very slow. COPY streams a whole table as one block of CSV text, and
PostgreSQL loads it in a single pass, typically 50 to 100 times faster.

This matters more on AWS than it did locally: the rows are created on your
machine and travel over the internet to Amazon RDS. At scale 0.2 that is about
140 MB in total, which takes a few minutes on a db.t4g.micro instance. One
COPY per table keeps that to 20 transfers instead of a million round trips.

WHY THE ORDER MATTERS
---------------------
Foreign keys make Postgres check, for every row, that what it points to
already exists. So a table can only be loaded after the tables it points
to. LOAD_ORDER below lists all 20 tables in a safe order.
"""

import io

import polars as pl
import psycopg

# Every table appears after all the tables it points to.
LOAD_ORDER = [
    "scheme_categories", "benchmarks", "benchmark_values", "schemes", "scheme_plans",
    "fund_managers", "scheme_manager_assignments", "nav_history",
    "sectors", "issuer_groups", "issuers", "securities", "credit_rating_history",
    "distributors", "investors", "folios", "sip_registrations",
    "transactions", "scheme_aum_monthly", "portfolio_holdings",
]


def load_all(database_url: str, tables: dict[str, pl.DataFrame]) -> None:
    """Empty all 20 tables, then load each DataFrame into its table."""

    # Guard: every table must have been generated before we delete anything.
    missing = set(LOAD_ORDER) - set(tables)
    if missing:
        raise ValueError(f"Not generated: {sorted(missing)}")

    # One connection, one transaction: if any table fails to load, the
    # whole load is undone and the database keeps its previous contents.
    with psycopg.connect(database_url) as conn:
        # TRUNCATE empties tables instantly. CASCADE is needed because the
        # tables point to each other; listing all 20 empties them together.
        conn.execute("TRUNCATE " + ", ".join(LOAD_ORDER) + " CASCADE")

        for name in LOAD_ORDER:
            frame = tables[name]
            print(f"  loading {name:<28} {len(frame):>10,} rows")

            # Write the DataFrame as CSV text into memory. Empty values
            # (None) become empty fields, which COPY reads as NULL.
            buffer = io.BytesIO()
            frame.write_csv(buffer, include_header=True)

            # Name the columns explicitly, so the CSV columns always match
            # the table columns, whatever order the DataFrame is in.
            columns = ", ".join(frame.columns)
            copy_sql = f"COPY {name} ({columns}) FROM STDIN WITH (FORMAT csv, HEADER true)"
            with conn.cursor() as cur, cur.copy(copy_sql) as copy:
                copy.write(buffer.getvalue())

        # Refresh PostgreSQL's statistics about each table (row counts, value
        # spreads). The query planner uses them to choose fast query plans, and
        # the metadata pipeline reads them to learn about each column: how many
        # different values it holds, how often it is empty, and its commonest
        # values. Without this, those notes would be drafted from nothing.
        conn.execute("ANALYZE")