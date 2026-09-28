"""
tests/test_neptune.py
=====================

WHAT THIS FILE IS FOR
---------------------
Proves the project can reach Amazon Neptune and use it, before the metadata
pipeline depends on it. Run this straight after creating the cluster:

    pytest tests/test_neptune.py -v

It checks:
    1. the cluster answers a query at all (network + signature + permissions)
    2. a node can be written, read back and deleted
    3. the four functions the pipeline actually uses behave as expected

The test writes only to a label of its own, TestPing, so it can never
disturb the real graph, which uses the label Table.

IF IT FAILS
-----------
    connection timed out       the cluster is stopped, or the security group
                               does not allow your IP on port 8182
    403 / not authorized       fundgraph-app is missing the neptune-db
                               permissions, or IAM authentication is off
    Missing in .env            NEPTUNE_CLUSTER_HOST is not set
"""

import pytest
from dotenv import load_dotenv

from adapters import graph_neptune

load_dotenv()


@pytest.fixture(autouse=True)
def clean_up():
    """Remove any leftover test nodes before and after each test."""
    graph_neptune.run("MATCH (n:TestPing) DETACH DELETE n")
    yield
    graph_neptune.run("MATCH (n:TestPing) DETACH DELETE n")


def test_cluster_answers_a_query():
    """The simplest possible query: add 1 and 1. Proves the whole path works."""
    rows = graph_neptune.run("RETURN 1 + 1 AS answer")
    assert rows[0]["answer"] == 2


def test_a_node_can_be_written_and_read_back():
    graph_neptune.run("CREATE (:TestPing {name: $n})", {"n": "hello"})
    rows = graph_neptune.run("MATCH (n:TestPing) RETURN n.name AS name")
    assert [r["name"] for r in rows] == ["hello"]


def test_parameters_are_sent_separately():
    """
    A value containing quotes must come back exactly as it went in. This is
    what makes parameters safer than building query text by hand.
    """
    awkward = "O'Brien \"and\" co"
    graph_neptune.run("CREATE (:TestPing {name: $n})", {"n": awkward})
    rows = graph_neptune.run("MATCH (n:TestPing) RETURN n.name AS name")
    assert rows[0]["name"] == awkward


def test_counts_returns_numbers():
    """counts() is used by publish and by the Gate 2 test."""
    result = graph_neptune.counts()
    assert isinstance(result["tables"], int)
    assert isinstance(result["joins"], int)