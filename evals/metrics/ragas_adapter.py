"""
evals/metrics/ragas_adapter.py
==============================

WHY THIS FILE EXISTS
--------------------
RAGAS is a widely used open-source library for scoring RAG systems. Clients
and reviewers recognise its metric names, so being able to say "we also score
with RAGAS" is worth something. This file translates our results into the
shape RAGAS expects, and runs it.

It is OPTIONAL and runs last:

    pip install ragas langchain-aws
    python -m evals.run_eval --ragas

WHAT IT EVALUATES
-----------------
RAGAS was built for document RAG: a question, some retrieved text, and an
answer. Our pipeline retrieves TABLES and runs SQL, so the parts are mapped
like this:

    question          the question, unchanged
    contexts          the table and column notes the SQL was written from,
                      plus the rows that came back. These are the "documents"
    answer            the explanation the agent wrote
    ground_truth      the tables and joins the question needed, as text

    faithfulness        is the explanation supported by those contexts?
    answer_relevancy    does the explanation address the question?

WHAT THE EXPECTED OUTPUT LOOKS LIKE
-----------------------------------
    RAGAS   faithfulness 0.88   answer_relevancy 0.91

WHAT IT MEANS, IN PLAIN WORDS
-----------------------------
Both run from 0 to 1, higher is better.

    faithfulness 0.88      about 88% of the statements in the explanations
                           could be traced back to the data shown. The rest
                           were extra wording the data did not support.
    answer_relevancy 0.91  the explanations mostly answered the question that
                           was asked, rather than a nearby one.

READ THESE WITH CARE, FOR THREE REASONS
---------------------------------------
1. RAGAS asks a model to score, so the numbers move a little between runs,
   even on the same answers.
2. It costs model calls on top of the evaluation itself.
3. It was designed for paragraphs of text, not for table rows, so it is
   less precise here than our own groundedness check, which looks for each
   number in the actual result.

Use it alongside the deterministic numbers, never instead of them. If RAGAS
is not installed, the evaluation prints a note and carries on.
"""

import os


def available() -> tuple[bool, str]:
    """Is RAGAS installed and usable? Returns (yes/no, reason if no)."""
    try:
        import ragas  # noqa: F401
        from langchain_aws import BedrockEmbeddings, ChatBedrockConverse  # noqa: F401
    except ImportError as error:
        return False, f"not installed ({error.name}). Install with: pip install ragas langchain-aws"
    return True, ""


def _bedrock_models():
    """
    Point RAGAS at the same Bedrock model and embeddings the project uses,
    with the same keys and region, so it is scoring like-for-like.
    """
    from langchain_aws import BedrockEmbeddings, ChatBedrockConverse

    from metadata.llm import aws_settings

    settings = aws_settings()
    common = {
        "region_name": settings["region"],
        "aws_access_key_id": settings["access_key"],
        "aws_secret_access_key": settings["secret_key"],
    }
    chat = ChatBedrockConverse(model=settings["model_id"], temperature=0, **common)
    embeddings = BedrockEmbeddings(model_id=os.environ.get("RAGAS_EMBEDDING_MODEL",
                                                           "amazon.titan-embed-text-v2:0"), **common)
    return chat, embeddings


def to_ragas_rows(runs: list[dict]) -> list[dict]:
    """
    Turn our per-question results into RAGAS's four fields.

    Only questions that completed are included: a refused question has no
    explanation to score, and counting it as 0 would be misleading.
    """
    rows = []
    for run in runs:
        result = run.get("result") or {}
        if result.get("status") != "completed" or not result.get("commentary"):
            continue

        # The "documents" our answer was written from: the notes used, and
        # the rows returned.
        contexts = []
        for table in result.get("tables") or []:
            note = (run.get("notes_used") or {}).get(table)
            contexts.append(f"{table}: {note}" if note else table)
        if result.get("sql"):
            contexts.append(f"SQL: {result['sql']}")
        query_result = result.get("result") or {}
        for row in (query_result.get("rows") or [])[:20]:
            contexts.append(" | ".join("" if v is None else str(v) for v in row))

        expected = run.get("expected") or {}
        rows.append({
            "question": run["question"],
            "contexts": contexts or ["(no context)"],
            "answer": result["commentary"],
            "ground_truth": "Tables: " + ", ".join(expected.get("tables") or []),
        })
    return rows


def evaluate(runs: list[dict]) -> dict:
    """
    Score the completed questions with RAGAS.

    Returns {"faithfulness": 0..1, "answer_relevancy": 0..1, "questions": n}
    or {"skipped": "<reason>"} if RAGAS is missing or something goes wrong.
    The evaluation never fails because of this: the numbers from our own
    checks are already in hand by the time this runs.
    """
    ok, reason = available()
    if not ok:
        return {"skipped": reason}

    rows = to_ragas_rows(runs)
    if not rows:
        return {"skipped": "no completed questions to score"}

    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import answer_relevancy, faithfulness

        chat, embeddings = _bedrock_models()
        scores = ragas_evaluate(
            Dataset.from_list(rows),
            metrics=[faithfulness, answer_relevancy],
            llm=chat,
            embeddings=embeddings,
        )
        summary = scores.to_pandas()[["faithfulness", "answer_relevancy"]].mean().to_dict()
        return {
            "faithfulness": round(float(summary.get("faithfulness", 0)), 3),
            "answer_relevancy": round(float(summary.get("answer_relevancy", 0)), 3),
            "questions": len(rows),
        }
    except Exception as error:
        return {"skipped": f"RAGAS failed: {str(error)[:200]}"}