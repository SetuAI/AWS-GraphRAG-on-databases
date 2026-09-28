"""
metadata/retrieval.py
=====================

WHAT THIS FILE IS FOR
---------------------
The three lookups the agent makes before writing SQL. Stage 6 calls these;
the Stage 5 tests call them to prove the metadata works.

    search_tables(question)       -> which tables the question is about      (pgvector)
    match_values(text)            -> exact database values for loose names   (pgvector)
    find_join_path(edges, tables) -> how to join those tables together       (Neptune)

HOW find_join_path WORKS
------------------------
The graph in Neptune has one node per table and one JOINS edge per approved
link. The edges are read ONCE, with adapters.graph_neptune.fetch_edges(),
and the route is then worked out here, in Python.

    Why not ask Neptune for the route? There are 20 tables and 22 links, so
    the whole graph is a few kilobytes. Reading it once and searching in
    memory means ONE request per question instead of one per pair of tables,
    and the search itself is instant. On a client database with hundreds of
    tables the same approach still holds: the graph describes the shape of
    the schema, not the rows, so it stays small.

    For two tables: find the route with the fewest joins. If two routes are
    equally short, prefer the one whose links we are most confident in.

    For more than two tables: start with the first, then repeatedly add
    whichever remaining table is closest to those already joined, with the
    route to it. That gives a small set of joins connecting everything the
    question needs.
"""

from metadata import llm


def _vector(question: str) -> str:
    """Embed one piece of text, written the way pgvector reads a vector."""
    return str(llm.embed([question])[0])


def search_tables(conn, question: str, k: int = 5) -> list[tuple[str, float]]:
    """
    The k tables whose notes are closest in meaning to the question.
    <=> is pgvector's cosine distance: 0 = same meaning, 2 = opposite.
    We return 1 - distance as a similarity score, where higher is better.
    """
    v = _vector(question)
    return conn.execute(
        """SELECT table_name, 1 - (embedding <=> %s::vector) AS score FROM meta_embeddings
           WHERE kind = 'table' ORDER BY embedding <=> %s::vector LIMIT %s""",
        (v, v, k),
    ).fetchall()


def match_values(conn, text: str, k: int = 3, table: str | None = None,
                 column: str | None = None) -> list[tuple[str, str, str, float]]:
    """
    The k stored database values closest to a name as a user typed it.
    Example: "sahyadri credit fund" -> ("schemes", "scheme_name", "Sahyadri Credit Risk Fund", 0.8x)
    Optionally limited to one table and column.
    Returns (table, column, value, similarity).

    Embeddings alone are fuzzy on short names: "vardhaman housing" sits as
    close to "Vardhaman Holdings Ltd" as to "Vardhaman Housing Finance Ltd".
    So pgvector fetches a wider pool, and the pool is re-ranked by how many
    of the typed words appear in each value, with the vector score as the
    tie-break.
    """
    v = _vector(text)
    pool = conn.execute(
        """SELECT table_name, column_name, content, 1 - (embedding <=> %s::vector) AS score
           FROM meta_embeddings
           WHERE kind = 'value' AND (%s::text IS NULL OR table_name = %s) AND (%s::text IS NULL OR column_name = %s)
           ORDER BY embedding <=> %s::vector LIMIT %s""",
        (v, table, table, column, column, v, max(k * 10, 30)),
    ).fetchall()
    typed = set(text.lower().split())

    def word_overlap(value: str) -> float:
        return len(typed & set(value.lower().split())) / len(typed) if typed else 0.0

    return sorted(pool, key=lambda row: (word_overlap(row[2]), row[3]), reverse=True)[:k]


def _neighbours(edges: list[dict]) -> dict[str, list[dict]]:
    """
    For each table, the edges that touch it.

    A join works in both directions: transactions -> folios is also
    folios -> transactions when you are looking for a route. So each edge is
    filed under both of its tables.
    """
    by_table: dict[str, list[dict]] = {}
    for edge in edges:
        by_table.setdefault(edge["from_table"], []).append(edge)
        by_table.setdefault(edge["to_table"], []).append(edge)
    return by_table


def _other_end(edge: dict, table: str) -> str:
    """The table at the other end of an edge from the one given."""
    return edge["to_table"] if edge["from_table"] == table else edge["from_table"]


def _route(by_table: dict[str, list[dict]], a: str, b: str, max_hops: int) -> list[dict] | None:
    """
    The best route from table a to table b, or None if there is none within
    max_hops joins.

    This is a breadth-first search: look at everything one join away, then
    everything two joins away, and so on. The first time b is reached, that
    is a shortest route, because nothing shorter was left unexplored.

    Among routes of the same length, the one with the highest total
    confidence wins: a declared foreign key (1.0) is preferred over a link
    that was discovered by matching values (less than 1.0).
    """
    if a == b:
        return []
    # Each item waiting to be explored: (table reached, the route taken to it).
    frontier = [(a, [])]
    seen = {a}
    for _ in range(max_hops):
        best_here: list[list[dict]] = []
        next_frontier = []
        for table, route in frontier:
            for edge in by_table.get(table, []):
                nxt = _other_end(edge, table)
                if nxt == b:
                    best_here.append(route + [edge])
                elif nxt not in seen:
                    next_frontier.append((nxt, route + [edge]))
        if best_here:
            # Shortest is guaranteed; pick the most confident among them.
            # The last part of the key just makes ties break the same way every run.
            return min(best_here, key=lambda r: (-sum(e["confidence"] for e in r),
                                                 [(e["from_table"], e["from_column"]) for e in r]))
        seen.update(t for t, _ in next_frontier)
        frontier = next_frontier
        if not frontier:
            break
    return None


def find_join_path(edges: list[dict], tables: list[str], max_hops: int = 6) -> dict:
    """
    The joins needed to connect all the given tables.

    edges:  every JOINS edge in the graph, from graph_neptune.fetch_edges()
    tables: the tables the question needs
    Returns {"tables": [...every table involved...], "joins": [edge, ...]},
    where each edge has from_table, from_column, to_table, to_column.
    Raises ValueError if some table cannot be reached within max_hops.
    """
    wanted = list(dict.fromkeys(tables))          # remove duplicates, keep order
    if not wanted:
        return {"tables": [], "joins": []}
    by_table = _neighbours(edges)
    joined = {wanted[0]}
    joins: list[dict] = []

    while any(t not in joined for t in wanted):
        # For every table still missing, find its best route to any table
        # already joined; then take the closest one.
        candidates = []
        for target in (t for t in wanted if t not in joined):
            for start in joined:
                route = _route(by_table, start, target, max_hops)
                if route:
                    candidates.append(route)
        if not candidates:
            missing = [t for t in wanted if t not in joined]
            raise ValueError(f"No join route within {max_hops} hops to: {missing}")
        route = min(candidates, key=lambda r: (len(r), -sum(e["confidence"] for e in r),
                                               [(e["from_table"], e["from_column"]) for e in r]))
        for edge in route:
            if edge not in joins:
                joins.append(edge)
            joined.update((edge["from_table"], edge["to_table"]))

    ordered = list(dict.fromkeys(wanted + [t for e in joins for t in (e["from_table"], e["to_table"])]))
    return {"tables": ordered, "joins": joins}