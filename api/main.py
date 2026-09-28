"""
api/main.py
===========

WHAT THIS FILE IS FOR
---------------------
The web API in front of the agent, and the demo page.

    GET  /               -> the demo page (api/static/index.html)
    POST /query          -> ask a question; returns table, commentary and every detail
    GET  /schema-graph   -> the 20 tables and their approved links, for drawing the graph
    GET  /graph          -> a full-screen, clickable view of that graph (api/static/graph.html)
    GET  /health         -> "ok" if the server is up
    GET  /docs           -> FastAPI's automatic page for trying the API by hand

HOW TO RUN
----------
    uvicorn api.main:app --port 8000
Then open http://localhost:8000
"""

import uuid

# Tracing is switched on BEFORE the app is created, so the agent's spans
# have somewhere to go from the very first request.
from api import telemetry

telemetry.setup()

from pathlib import Path  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from agent import context  # noqa: E402
from agent.graph import ask  # noqa: E402

app = FastAPI(title="FundGraph: ask the mutual fund database")
telemetry.instrument(app)   # record every HTTP request as a trace (AWS X-Ray)
STATIC = Path(__file__).parent / "static"


class QueryRequest(BaseModel):
    """What the caller sends. conversation_id ties follow-up questions together."""
    question: str
    conversation_id: str | None = None


@app.post("/query")
def query(request: QueryRequest) -> dict:
    # No ID, an empty one, or the text "null" (easy to type by mistake in
    # /docs) all mean: start a new conversation.
    given = (request.conversation_id or "").strip()
    conversation_id = given if given and given.lower() != "null" else str(uuid.uuid4())
    state = ask(request.question, conversation_id)
    result = state.get("result") or {}
    understanding = state.get("understanding") or {}
    # Show the tables the SQL really used, and only the joins between them.
    # If no SQL was checked, fall back to the tables the agent considered.
    path = state.get("join_path") or {}
    used = state.get("sql_tables") or path.get("tables", state.get("tables", []))
    joins = [j for j in path.get("joins", []) if j["from_table"] in used and j["to_table"] in used]
    return {
        "status": state.get("status"),
        "conversation_id": conversation_id,
        "message": state.get("block_reason") or (understanding.get("clarification_question")
                                                 if state.get("status") == "needs_clarification" else None),
        "tables": [{"columns": result.get("columns", []), "rows": result.get("rows", [])[:500]}] if result else [],
        "commentary": state.get("commentary"),
        "details": {
            "standalone_question": understanding.get("standalone_question"),
            "intent": understanding.get("intent"),
            "time_range": understanding.get("time_range"),
            "metrics_used": understanding.get("metrics", []),
            "entities": state.get("entities", []),
            "tables_considered": state.get("tables", []),
            "tables_used": used,
            "join_path": joins,
            "sql": state.get("sql"),
            "sql_explanation": state.get("sql_explanation"),
            "checks": state.get("checks", []),
            "attempts": state.get("attempts", 0),
            "input_guard": state.get("input_guard"),
            "steps": state.get("steps", []),
        },
    }


@app.get("/schema-graph")
def schema_graph() -> dict:
    """
    Nodes and edges for drawing the graph, read from agent_meta.
      nodes  -> table names (used by the demo page)
      tables -> per table: description, grain and estimated rows (used by /graph)
      edges  -> approved links, with their join columns and how each was found
    """
    with context.agent_conn() as conn:
        edges = conn.execute(
            """SELECT from_table, from_column, to_table, to_column, source FROM meta_edges
               WHERE status = 'approved' ORDER BY edge_id""").fetchall()
        tables = conn.execute(
            """SELECT table_name, description, grain, row_estimate FROM meta_tables
               WHERE status = 'approved' ORDER BY table_name""").fetchall()
    return {
        "nodes": [t[0] for t in tables],
        "tables": {t[0]: {"description": t[1], "grain": t[2], "rows": t[3]} for t in tables},
        "edges": [dict(zip(["from_table", "from_column", "to_table", "to_column", "source"], e)) for e in edges],
    }


@app.get("/graph")
def graph_page():
    """Full-screen, clickable view of the schema graph."""
    return FileResponse(STATIC / "graph.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/")
def page():
    return FileResponse(STATIC / "index.html")