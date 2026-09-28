"""
metadata/store.py
=================

WHAT THIS FILE IS FOR
---------------------
Creates and describes the tables in agent_meta that hold everything the
pipeline learns about the fund database. This is the "control plane" from
tradeoff 1: the agent's own data lives here, separate from the fund data.

    meta_tables      -> one note per fund table: what it holds, one row per what
    meta_columns     -> one note per column: what it means, whether it is PII
    meta_edges       -> every link between tables, where it came from, how sure we are
    meta_embeddings  -> the pgvector store: numbers for table notes and category values

The graph itself lives in Amazon Neptune, which publish.py rebuilds from
meta_edges. See adapters/graph_neptune.py.

STATUS: WHY EVERYTHING STARTS UNAPPROVED
----------------------------------------
Notes start as 'draft' and inferred links as 'proposed'. Only a human
review (metadata/review.py) turns them 'approved', and publish.py only ever
uses approved items. An LLM-written note or a guessed link never reaches the
SQL-writing model without a person having looked at it.

meta_edges is the SOURCE OF TRUTH for links. The Neptune graph is a copy built
from it, so it can be deleted and rebuilt at any time without losing anything.
"""

import psycopg

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta_tables (
    table_name      text PRIMARY KEY,
    -- A hash of the table's columns and types. If it has not changed since
    -- the last run, the note is still valid and is not redrafted (saves LLM calls).
    fingerprint     text NOT NULL,
    row_estimate    bigint NOT NULL,
    description     text,
    grain           text,              -- "one row per ..." : what a single row represents
    business_terms  text[],            -- words users might say that point to this table
    status          text NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'approved')),
    drafted_at      timestamptz,
    approved_at     timestamptz
);

CREATE TABLE IF NOT EXISTS meta_columns (
    table_name      text NOT NULL REFERENCES meta_tables (table_name) ON DELETE CASCADE,
    column_name     text NOT NULL,
    data_type       text NOT NULL,
    is_pii          boolean NOT NULL DEFAULT false,
    distinct_values double precision,
    description     text,
    PRIMARY KEY (table_name, column_name)
);

CREATE TABLE IF NOT EXISTS meta_edges (
    edge_id         bigserial PRIMARY KEY,
    from_table      text NOT NULL,
    from_column     text NOT NULL,
    to_table        text NOT NULL,
    to_column       text NOT NULL,
    -- How the link was found:
    --   declared      -> a foreign key in the database itself
    --   same_name     -> column name matches another table's key, values checked
    --   value_overlap -> names differ, but the stored values match
    source          text NOT NULL CHECK (source IN ('declared', 'same_name', 'value_overlap')),
    confidence      numeric(4, 3) NOT NULL,     -- 1.000 = certain
    evidence        text,                       -- plain-English reason, shown in review
    status          text NOT NULL DEFAULT 'proposed' CHECK (status IN ('proposed', 'approved', 'rejected')),
    UNIQUE (from_table, from_column, to_table, to_column)
);

CREATE TABLE IF NOT EXISTS meta_embeddings (
    embedding_id    bigserial PRIMARY KEY,
    -- 'table' = a table's note; 'value' = one category value, e.g. a scheme name
    kind            text NOT NULL CHECK (kind IN ('table', 'value')),
    table_name      text NOT NULL,
    column_name     text,                        -- set for 'value' rows only
    content         text NOT NULL,               -- the text that was embedded
    -- 1,024 numbers from Titan Text Embeddings V2 (config/models.yaml).
    embedding       vector(1024) NOT NULL
);

-- HNSW is pgvector's fast search index. vector_cosine_ops means "closest"
-- is measured by cosine distance, which is what Titan embeddings are built for.
CREATE INDEX IF NOT EXISTS meta_embeddings_hnsw
    ON meta_embeddings USING hnsw (embedding vector_cosine_ops);
"""


def connect(agent_db_url: str) -> psycopg.Connection:
    """
    Open a connection to agent_meta on Amazon RDS.

    autocommit=True saves each statement as it runs, which is what the
    metadata steps want: if a long build stops halfway, the tables drafted so
    far are kept, and the next run only redoes what is missing.
    """
    return psycopg.connect(agent_db_url, autocommit=True, connect_timeout=15)


def ensure_schema(conn: psycopg.Connection) -> None:
    """Create the four tables and the index if they do not exist yet. Safe to run repeatedly."""
    conn.execute(SCHEMA_SQL)