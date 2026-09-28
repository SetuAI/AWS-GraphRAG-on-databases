"""
adapters/graph_neptune.py
=========================

WHAT THIS FILE IS FOR
---------------------
The ONE place in the project that talks to Amazon Neptune, the graph
database that stores how the 20 tables connect. Everything else calls one
of these four functions:

    rebuild(tables, edges)   wipe the graph and write it again from scratch
    fetch_edges()            read every join edge back out
    counts()                 how many tables and joins are stored
    run(query, parameters)   send any openCypher query (used by the three above)

WHY NEPTUNE, AND NOT POSTGRES
-----------------------------
On the Azure version the graph lived inside Postgres, using an extension
called Apache AGE. Amazon RDS does not offer AGE, so the graph moved to
Neptune, which is AWS's own graph database. Nothing else about the design
changed: meta_edges in agent_meta is still the SOURCE OF TRUTH for links,
and this graph is a copy of it, rebuilt by metadata/publish.py.

HOW IT CONNECTS
---------------
Neptune has no driver to install and no password. You send openCypher
queries as ordinary HTTPS POST requests to:

    https://<cluster endpoint>:8182/opencypher

and every request must be SIGNED with your AWS keys (SigV4), the same way
the AWS CLI signs its own calls. The signature IS the authentication, which
is why the cluster was created with IAM database authentication switched on.

WHAT IT READS FROM .env
-----------------------
    NEPTUNE_CLUSTER_HOST   the cluster's writer address, ending in
                           .ap-south-1.neptune.amazonaws.com
    NEPTUNE_PORT           8182
    AWS_ACCESS_KEY         the keys the request is signed with
    AWS_SECRET_KEY
    AWS_BEDROCK_ARN        the region is read from this (see metadata/llm.py)

IF IT FAILS
-----------
    connection timed out        the cluster is stopped, or the security group
                                does not allow your IP address on port 8182
    403 / signature errors      the keys are wrong, or fundgraph-app is missing
                                the neptune-db permissions
    "Missing in .env"           NEPTUNE_CLUSTER_HOST is not set
"""

import json
import os

import requests
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials
from dotenv import load_dotenv

# The region and the AWS keys are worked out once, in metadata/llm.py, from
# .env. Reusing that keeps every AWS setting in one place.
from metadata.llm import aws_settings

# How long to wait for Neptune to answer, in seconds. The graph is tiny, so a
# slow answer means a network problem, not a slow query.
TIMEOUT_SECONDS = 30


def _endpoint() -> str:
    """The full address to POST openCypher queries to."""
    load_dotenv()
    host = os.environ.get("NEPTUNE_CLUSTER_HOST", "").strip()
    if not host:
        raise RuntimeError("Missing in .env: NEPTUNE_CLUSTER_HOST (see .env.example)")
    port = os.environ.get("NEPTUNE_PORT", "8182").strip() or "8182"
    return f"https://{host}:{port}/opencypher"


def run(query: str, parameters: dict | None = None) -> list[dict]:
    """
    Send one openCypher query to Neptune and return its rows.

    PARAMETERS, NOT STRING BUILDING
    -------------------------------
    Values are passed separately, as parameters, and referred to in the query
    as $name:

        run("MATCH (t:Table {name: $n}) RETURN t", {"n": "schemes"})

    Neptune keeps the value and the query apart, so a table name can never be
    read as part of the query itself. (The Azure version had to build query
    text by hand, because AGE does not accept parameters.)
    """
    settings = aws_settings()
    body = {"query": query}
    if parameters:
        # Neptune expects the parameters as one JSON string in the form body.
        body["parameters"] = json.dumps(parameters)

    # Build the request, then sign it. SigV4Auth adds an Authorization header
    # that proves the request came from the holder of these keys and has not
    # been altered. Neptune checks it against the IAM policy on fundgraph-app.
    aws_request = AWSRequest(
        method="POST",
        url=_endpoint(),
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    SigV4Auth(
        Credentials(settings["access_key"], settings["secret_key"]),
        "neptune-db",                 # the service name Neptune signs as
        settings["region"],
    ).add_auth(aws_request)

    response = requests.post(
        _endpoint(),
        data=body,
        headers=dict(aws_request.headers),
        timeout=TIMEOUT_SECONDS,
        # Neptune presents an AWS certificate; verify=True is the default and
        # is left on. Never switch this off to "make it work".
    )
    if response.status_code != 200:
        raise RuntimeError(f"Neptune returned {response.status_code}: {response.text[:300]}")

    # A successful answer looks like {"results": [ {...}, {...} ]}
    return response.json().get("results", [])


def rebuild(tables: list[dict], edges: list[dict]) -> dict:
    """
    Replace the whole graph with the given tables and joins.

    tables: [{"name": "schemes", "grain": "One row per scheme"}, ...]
    edges:  [{"from_table": ..., "from_column": ..., "to_table": ...,
              "to_column": ..., "source": ..., "confidence": 1.0}, ...]

    Wiping and rewriting is the simplest thing that is always correct: the
    graph then matches meta_edges exactly, with no leftovers from a link that
    was rejected since the last run. It is only a few dozen rows.
    """
    # 1. Empty the graph. DETACH DELETE removes each node and its edges.
    run("MATCH (n) DETACH DELETE n")

    # 2. Create one node per table. UNWIND takes a list of values and repeats
    #    the CREATE for each one, so this is a single request rather than 20.
    run(
        "UNWIND $rows AS row CREATE (:Table {name: row.name, grain: row.grain})",
        {"rows": [{"name": t["name"], "grain": t.get("grain") or ""} for t in tables]},
    )

    # 3. Create one JOINS edge per approved link. MATCH finds the two table
    #    nodes by name, then CREATE joins them and stores the columns on the
    #    edge itself, which is what the SQL step needs later.
    if edges:
        run(
            """
            UNWIND $rows AS row
            MATCH (a:Table {name: row.from_table}), (b:Table {name: row.to_table})
            CREATE (a)-[:JOINS {
                from_table: row.from_table, from_column: row.from_column,
                to_table:   row.to_table,   to_column:   row.to_column,
                source:     row.source,     confidence:  row.confidence
            }]->(b)
            """,
            {"rows": [
                {
                    "from_table": e["from_table"], "from_column": e["from_column"],
                    "to_table": e["to_table"], "to_column": e["to_column"],
                    "source": e.get("source", "declared"),
                    "confidence": float(e.get("confidence", 1.0)),
                }
                for e in edges
            ]},
        )
    return counts()


def fetch_edges() -> list[dict]:
    """
    Every join edge in the graph, as plain dictionaries.

    The agent reads all 22 edges once per question and works out the join
    route in Python (see metadata/retrieval.py). One small request beats
    sending a path query to Neptune for every pair of tables.
    """
    rows = run(
        """
        MATCH (a:Table)-[j:JOINS]->(b:Table)
        RETURN j.from_table AS from_table, j.from_column AS from_column,
               j.to_table   AS to_table,   j.to_column   AS to_column,
               j.source     AS source,     j.confidence  AS confidence
        """
    )
    return [dict(row) for row in rows]


def fetch_tables() -> list[dict]:
    """Every table node in the graph: its name and what one row means."""
    rows = run("MATCH (t:Table) RETURN t.name AS name, t.grain AS grain ORDER BY t.name")
    return [dict(row) for row in rows]


def counts() -> dict:
    """How many table nodes and join edges are stored. Used by tests and publish."""
    tables = run("MATCH (t:Table) RETURN count(t) AS n")[0]["n"]
    joins = run("MATCH ()-[j:JOINS]->() RETURN count(j) AS n")[0]["n"]
    return {"tables": tables, "joins": joins}