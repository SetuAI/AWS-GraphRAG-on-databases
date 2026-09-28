"""
tests/test_rds_setup.py
=======================

WHAT THIS FILE IS FOR
---------------------
The first test to run after creating the RDS instance and running
scripts/rds_setup.py. It proves the database side is ready, before any data,
metadata or agent work begins:

    1. mf_data is reachable with the settings in .env
    2. agent_meta is reachable
    3. both are on the SAME RDS instance (one instance, two databases)
    4. pgvector is switched on in agent_meta, and can store a vector
    5. PostgreSQL is version 15 or newer, which pgvector needs
    6. Apache AGE is NOT there, which is expected on RDS: the graph lives in
       Amazon Neptune instead

If any of these fail, nothing later can work, and the failure message says
which setting or step to fix.

HOW TO RUN
----------
    pytest tests/test_rds_setup.py -v
"""

import os
from urllib.parse import urlparse

import psycopg
import pytest
from dotenv import load_dotenv

# Read DATA_DB_URL and AGENT_DB_URL from .env before any test runs.
load_dotenv()


def connect(url_setting: str) -> psycopg.Connection:
    """Open a connection using one of the URL settings from .env."""
    return psycopg.connect(os.environ[url_setting], connect_timeout=15, autocommit=True)


@pytest.fixture(scope="module")
def data_db():
    """A connection to mf_data, shared by the tests that need it."""
    with connect("DATA_DB_URL") as conn:
        yield conn


@pytest.fixture(scope="module")
def agent_db():
    """A connection to agent_meta, shared by the tests that need it."""
    with connect("AGENT_DB_URL") as conn:
        yield conn


# ---------------------------------------------------------------------------
# 1 and 2: both databases answer
# ---------------------------------------------------------------------------

def test_fund_database_is_reachable(data_db):
    # "SELECT 1" is the smallest possible query: it proves the connection
    # works without depending on any table existing.
    assert data_db.execute("SELECT 1").fetchone()[0] == 1


def test_agent_database_is_reachable(agent_db):
    assert agent_db.execute("SELECT 1").fetchone()[0] == 1


# ---------------------------------------------------------------------------
# 3: one RDS instance, two databases
# ---------------------------------------------------------------------------

def test_both_databases_are_on_the_same_instance():
    """
    The two URLs should differ only in the database name at the end. If the
    hosts differ, you are paying for two instances by mistake, or one URL
    still points at an old server.
    """
    data = urlparse(os.environ["DATA_DB_URL"])
    agent = urlparse(os.environ["AGENT_DB_URL"])
    assert data.hostname == agent.hostname, "DATA_DB_URL and AGENT_DB_URL are on different servers"
    assert data.path != agent.path, "Both URLs point at the SAME database; they must differ"


# ---------------------------------------------------------------------------
# 4 and 5: pgvector works in agent_meta
# ---------------------------------------------------------------------------

def test_pgvector_is_installed_in_agent_database(agent_db):
    installed = agent_db.execute(
        "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
    ).fetchone()
    assert installed is not None, "pgvector is not switched on: run python scripts/rds_setup.py"


def test_pgvector_can_store_and_compare_a_vector(agent_db):
    """
    Proves pgvector really works, not just that it is listed. It measures the
    distance between two 3-number vectors; anything above 0 means they differ.
    (The real embeddings are 1,024 numbers; the idea is the same.)
    """
    distance = agent_db.execute("SELECT '[1,2,3]'::vector <-> '[3,2,1]'::vector").fetchone()[0]
    assert distance > 0


def test_postgres_version_is_15_or_newer(agent_db):
    # server_version_num is the version as one number: 150004 = 15.4,
    # 170002 = 17.2. pgvector needs 15 or newer on RDS.
    version = int(agent_db.execute("SHOW server_version_num").fetchone()[0])
    assert version >= 150000, f"PostgreSQL {version} is too old for pgvector on RDS"


# ---------------------------------------------------------------------------
# 6: AGE is absent, as expected on RDS
# ---------------------------------------------------------------------------

def test_apache_age_is_not_switched_on(agent_db):
    """
    Not a problem: a reminder in test form.

    On the Azure build, the schema graph lived in Apache AGE inside Postgres.
    Amazon RDS does not offer AGE, so on AWS the graph lives in Amazon Neptune
    instead, and nothing here should depend on AGE being present.

    This checks that AGE is not switched on in agent_meta, which is what the
    rest of the code assumes.
    """
    switched_on = agent_db.execute(
        "SELECT count(*) FROM pg_extension WHERE extname = 'age'"
    ).fetchone()[0]
    assert switched_on == 0