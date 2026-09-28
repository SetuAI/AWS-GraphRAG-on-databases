"""
evals/metrics/retrieval.py
==========================

WHY WE DO THIS
--------------
Before the agent can answer anything, it has to find the right tables and
work out how to join them. If it picks the wrong tables, nothing later can
save the answer: the SQL will be about the wrong thing. So this is measured
separately from the answer itself, and it is measured first.

WHAT WE ARE EVALUATING
----------------------
The FINDING part of the pipeline: steps 4 and 5, where tables are chosen and
the join route is found through the Neptune graph. These are the standard RAG
retrieval metrics, written for a database instead of documents:

    precision@k   of the tables it chose, how many were actually needed
    recall@k      of the tables needed, how many it chose
    f1            the two combined into one number
    mrr           how near the top the first correct table came
    join_path     were the required join conditions used

WHY BOTH PRECISION AND RECALL
-----------------------------
They pull against each other, and one alone is easy to cheat.

    Choose all 20 tables       -> recall 1.00, precision 0.25. The SQL model is
                                  buried in tables it does not need.
    Choose one certain table   -> precision 1.00, recall 0.33. The answer is
                                  missing the tables that hold the rest.

Recall matters more here: a missing table means the question cannot be
answered at all, while a spare table usually just adds noise. That is why
the report shows both, and never averages them into a single score.

WHAT COUNTS AS "NEEDED"
-----------------------
The tables listed for that question in evals/questions/questions.yaml,
written by hand from the data. Never taken from what the agent answered.

WHAT THE EXPECTED OUTPUT LOOKS LIKE
-----------------------------------
For one question:

    precision@5 0.75   recall@5 1.00   f1 0.86   MRR 1.00
    join_path   {"required": 2, "matched": 2, "exact": true, "missing": []}

WHAT IT MEANS, IN PLAIN WORDS
-----------------------------
    precision@5 0.75   it chose 4 tables and 3 of them were needed. The spare
                       one adds noise to the SQL prompt, but does no harm.
    recall@5 1.00      it found every table the question needed. Nothing is
                       missing, so a correct answer is possible.
    f1 0.86            the two above, combined. Low unless both are decent.
    MRR 1.00           the very first table it chose was one of the needed
                       ones, so its ordering is sensible.
    join_path exact    it used both join conditions the question required. For
                       a question that needs a HIDDEN link, this can only be
                       true if the pipeline discovered that link by itself.

    A score of 0 on recall means the question could not have been answered.
    A score of 0 on precision with high recall means it grabbed too much.

EVERY FUNCTION HERE IS PLAIN ARITHMETIC. No model is called, so the same run
gives the same numbers every time.
"""

import re


def precision_at_k(selected: list[str], needed: list[str], k: int = 5) -> float:
    """
    Of the first k tables the agent selected, the share that were needed.

        selected = [transactions, scheme_plans, schemes, folios]
        needed   = [transactions, scheme_plans, schemes]
        -> 3 of 4 -> 0.75
    """
    top = selected[:k]
    if not top:
        return 0.0
    hits = sum(1 for table in top if table in set(needed))
    return hits / len(top)


def recall_at_k(selected: list[str], needed: list[str], k: int = 5) -> float:
    """
    Of the tables that were needed, the share the agent found in its first k.

        selected = [transactions, scheme_plans]
        needed   = [transactions, scheme_plans, schemes]
        -> 2 of 3 -> 0.67
    """
    if not needed:
        # Nothing was needed (a refusal question): treat as fully correct, so
        # refusals do not drag the average down.
        return 1.0
    top = set(selected[:k])
    return sum(1 for table in needed if table in top) / len(needed)


def f1_at_k(selected: list[str], needed: list[str], k: int = 5) -> float:
    """
    Precision and recall combined (their harmonic mean), which is low unless
    BOTH are high. 0 if either is 0.
    """
    p = precision_at_k(selected, needed, k)
    r = recall_at_k(selected, needed, k)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def reciprocal_rank(selected: list[str], needed: list[str]) -> float:
    """
    1 divided by the position of the first needed table.

        first in the list  -> 1.00
        second             -> 0.50
        third              -> 0.33
        not found          -> 0.00

    Averaged over all questions this is MRR. It answers a different question
    from recall: not "did it find them?" but "did it put them first?".
    """
    if not needed:
        return 1.0
    wanted = set(needed)
    for position, table in enumerate(selected, start=1):
        if table in wanted:
            return 1 / position
    return 0.0


def _normalise(condition: str) -> frozenset:
    """
    One join condition, in a form where the two sides can be compared in any
    order. "a.x = b.y" and "b.y = a.x" are the same join, so both become
    the same pair of names.
    """
    left, _, right = condition.partition("=")
    return frozenset({left.strip().lower(), right.strip().lower()})


def _join_used_in_sql(sql: str, condition: str) -> bool:
    """
    Does the SQL actually join on these two columns?

    The graph route and the SQL can differ. The route is a suggestion; the
    SQL is what ran. In this real example the route wandered through folios,
    while the SQL joined transactions to scheme_plans directly, which is
    correct. Scoring only the route would mark a right answer wrong.

    SQL uses table aliases ("JOIN scheme_plans sp ON t.plan_id = sp.plan_id"),
    so the table names are not there to match on. The COLUMN pair is, and for
    this schema a column pair identifies the join unambiguously.
    """
    if not sql:
        return False
    left, _, right = condition.partition("=")
    a = left.strip().split(".")[-1]
    b = right.strip().split(".")[-1]
    text = sql.lower()
    # "t.plan_id = sp.plan_id" in either order, with any aliases.
    for first, second in ((a, b), (b, a)):
        if re.search(rf"\b{re.escape(first)}\s*=\s*[\w\"]+\.{re.escape(second)}\b", text):
            return True
    return False


def join_path_score(used_joins: list[dict], required: list[str], sql: str = "") -> dict:
    """
    Did the answer use the join conditions the question needs?

    A required join counts as used if it is EITHER in the route the graph
    found OR in the SQL that ran. Both are reported separately, because they
    say different things:

        in_route  the graph could offer the join      -> the metadata is right
        in_sql    the query actually used it          -> the answer is right

    A question needing a HIDDEN link can only score here if that link was
    discovered in the first place, which makes this the most telling number
    in the report.

    Returns {"required", "matched", "exact", "missing", "in_route", "in_sql"}.
    """
    if not required:
        return {"required": 0, "matched": 0, "exact": True, "missing": [],
                "in_route": 0, "in_sql": 0}

    route = {
        _normalise(f"{j['from_table']}.{j['from_column']} = {j['to_table']}.{j['to_column']}")
        for j in used_joins or []
    }

    missing, in_route, in_sql = [], 0, 0
    for condition in required:
        by_route = _normalise(condition) in route
        by_sql = _join_used_in_sql(sql, condition)
        in_route += by_route
        in_sql += by_sql
        if not (by_route or by_sql):
            missing.append(condition)

    return {
        "required": len(required),
        "matched": len(required) - len(missing),
        "exact": not missing,
        "missing": missing,
        "in_route": in_route,
        "in_sql": in_sql,
    }


def score(result: dict, expected: dict, k: int = 5) -> dict:
    """
    Every retrieval number for one question.

    result:   what agent.graph.ask() returned
    expected: that question's block from questions.yaml
    """
    selected = result.get("tables") or []
    needed = expected.get("tables") or []
    joins = (result.get("join_path") or {}).get("joins") or []

    return {
        "precision_at_k": round(precision_at_k(selected, needed, k), 3),
        "recall_at_k": round(recall_at_k(selected, needed, k), 3),
        "f1_at_k": round(f1_at_k(selected, needed, k), 3),
        "reciprocal_rank": round(reciprocal_rank(selected, needed), 3),
        "join_path": join_path_score(joins, expected.get("joins") or [], result.get("sql") or ""),
        "tables_selected": selected,
        "tables_needed": needed,
    }