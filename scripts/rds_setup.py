"""
scripts/rds_setup.py
====================

WHAT THIS FILE IS FOR
---------------------
Prepares a fresh Amazon RDS for PostgreSQL instance for this project. It is
the first thing you run after the RDS instance finishes creating, and you
normally run it once.

It does three things:

    1. Creates the second database, agent_meta, if it is not there yet.
       (RDS gives you ONE database when the instance is created. This project
       needs two.)
    2. Switches on the pgvector extension inside agent_meta, so embeddings
       can be stored and searched there.
    3. Prints a short report of both databases, so you can see it worked.

WHY TWO DATABASES
-----------------
    mf_data     the fund data: 20 tables. Stands in for the client's own
                database. The agent only ever READS it.
    agent_meta  the agent's own data: table notes, links, and the pgvector
                embeddings. The agent reads and writes it.

Keeping them apart means the agent's write access never reaches fund data,
and a real client database never needs new tables or extensions added to it.

WHAT IS DIFFERENT FROM THE DOCKER VERSION
-----------------------------------------
With Docker, a start-up script created agent_meta and switched on both
pgvector and Apache AGE automatically. On RDS:

    - nothing is automatic, so this script does the same work, and
    - Apache AGE is NOT available on RDS at all. The schema graph therefore
      lives in Amazon Neptune instead. This script does not try to create it.

HOW TO RUN
----------
    python scripts/rds_setup.py

It reads two settings from .env:

    DATA_DB_URL   postgresql://user:password@your-db...rds.amazonaws.com:5432/mf_data
    AGENT_DB_URL  postgresql://user:password@your-db...rds.amazonaws.com:5432/agent_meta

Both point at the SAME RDS instance; only the database name at the end differs.

IF IT FAILS
-----------
    could not connect            the instance is still creating, is not set to
                                 publicly accessible, or its security group does
                                 not allow your IP address on port 5432
    password authentication      the user or password in the URL is wrong
    permission denied to create  the user in DATA_DB_URL is not the master user
                                 created with the RDS instance
    extension "vector" is not    pgvector is missing on very old engine versions;
    available                    use PostgreSQL 15 or newer on RDS
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------

# "os" reads settings (environment variables) such as DATA_DB_URL.
import os

# urlparse splits a connection URL into its parts (user, password, host,
# port, database name), so we can swap the database name and keep the rest.
from urllib.parse import urlparse, urlunparse

# psycopg is the PostgreSQL driver: it opens connections and runs SQL.
import psycopg

# Reads the .env file, so os.environ contains DATA_DB_URL and AGENT_DB_URL.
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def database_name(url: str) -> str:
    """
    The database name at the end of a connection URL.

    urlparse("postgresql://u:p@host:5432/agent_meta").path is "/agent_meta",
    so removing the leading "/" leaves "agent_meta".
    """
    name = urlparse(url).path.lstrip("/")
    if not name:
        raise ValueError(f"No database name at the end of the URL: {url!r}")
    return name


def url_for_database(url: str, name: str) -> str:
    """
    The same connection URL, pointing at a different database on the same server.

    Used to connect to the built-in "postgres" database, because a database
    cannot be created from inside itself: you must be connected somewhere else
    while you create it.
    """
    parts = urlparse(url)
    # _replace returns a copy with one field changed; urlunparse rebuilds the URL.
    return urlunparse(parts._replace(path=f"/{name}"))


def connect(url: str) -> psycopg.Connection:
    """
    Open a connection where each statement is saved as soon as it runs
    (autocommit). CREATE DATABASE requires this: PostgreSQL refuses to run it
    inside a transaction.
    """
    return psycopg.connect(url, autocommit=True, connect_timeout=15)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    load_dotenv()

    # Square brackets: if a setting is missing, Python stops here with a clear
    # error naming it, rather than failing later with a vaguer message.
    data_url = os.environ["DATA_DB_URL"]
    agent_url = os.environ["AGENT_DB_URL"]

    data_db = database_name(data_url)
    agent_db = database_name(agent_url)
    host = urlparse(data_url).hostname
    print(f"RDS instance: {host}")
    print(f"  fund data  : {data_db}")
    print(f"  agent data : {agent_db}")

    # ---- 1. Check the instance is reachable, and which database exists ------
    # We connect to the built-in "postgres" database, which every PostgreSQL
    # server has. From there we can see and create other databases.
    admin_url = url_for_database(data_url, "postgres")
    with connect(admin_url) as conn:
        version = conn.execute("SELECT version()").fetchone()[0]
        print(f"\nConnected. Server: {version.split(',')[0]}")

        # pg_database lists every database on the server.
        existing = {row[0] for row in conn.execute("SELECT datname FROM pg_database").fetchall()}

        if data_db not in existing:
            # The fund database is created with the RDS instance itself, so if
            # it is missing, the URL is pointing at the wrong place.
            raise SystemExit(
                f"Database {data_db!r} does not exist on this instance. "
                f"Create the RDS instance with initial database name {data_db!r}, "
                f"or correct DATA_DB_URL in .env."
            )

        if agent_db in existing:
            print(f"Database {agent_db!r}: already exists, leaving it as it is.")
        else:
            # The database name cannot be passed as a parameter (%s) in
            # CREATE DATABASE, so it is placed in the text directly. It comes
            # from your own .env, and is checked below to contain only safe
            # characters, so nothing unexpected can be inserted here.
            if not agent_db.replace("_", "").isalnum():
                raise SystemExit(f"Unsafe database name in AGENT_DB_URL: {agent_db!r}")
            conn.execute(f'CREATE DATABASE "{agent_db}"')
            print(f"Database {agent_db!r}: created.")

    # ---- 2. Switch on pgvector inside agent_meta ---------------------------
    # An extension is switched on per database, not per server, so this must
    # run while connected to agent_meta itself.
    with connect(agent_url) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        installed = conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
        print(f"Extension 'vector' in {agent_db}: version {installed[0]}")

    # ---- 3. Report ---------------------------------------------------------
    # A quick look at both databases, so you can see the starting point:
    # mf_data is empty until apply_schema.py runs, and agent_meta is empty
    # until metadata.build runs.
    for label, url in (("mf_data", data_url), ("agent_meta", agent_url)):
        with connect(url) as conn:
            tables = conn.execute(
                "SELECT count(*) FROM pg_tables WHERE schemaname = 'public'"
            ).fetchone()[0]
            print(f"{label}: {tables} table(s) in the public schema")

    print("\nDone. Next: python dataplane/schema/apply_schema.py")


# This runs only when the file is run directly, not when another file imports it.
if __name__ == "__main__":
    main()