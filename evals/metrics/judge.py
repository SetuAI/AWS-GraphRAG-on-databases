"""
evals/metrics/judge.py
======================

WHY WE DO THIS
--------------
Arithmetic can prove a number was invented, but not that a sentence is
misleading, vague, or quietly claims that one thing caused another. For that
you need judgement, so this asks a model to read the answer and score it.

WHAT WE ARE EVALUATING
----------------------
The wording of the explanation. An optional second opinion: the chat model on Bedrock reads the question,
the rows and the explanation, and scores the explanation 1 to 5 on three
things arithmetic cannot see:

    faithfulness   does it only say what the rows support?
    relevancy      does it answer the question that was asked?
    clarity        would a business person understand it without help?

    python -m evals.run_eval --judge

WHY THIS IS OPTIONAL, AND SECOND
--------------------------------
The deterministic checks in generation.py come first, because they catch the
failure that matters most, an invented number, and they give the same answer
every time. This file adds judgement, and judgement wobbles:

  - Scores drift. The same explanation can score 4 today and 3 tomorrow.
  - It costs a model call per question, on top of the run itself.
  - The judge here is the same family of model that wrote the explanation,
    so it is not a neutral referee. Treat its scores as a hint, and never
    report them as accuracy.

That is why the report keeps these in their own column rather than mixing
them into the main numbers.

WHAT THE EXPECTED OUTPUT LOOKS LIKE
-----------------------------------
For one question:

    {"faithfulness": 5, "relevancy": 4, "clarity": 5,
     "reason": "States only what the rows show; mentions the fund but not the month."}

and in the report, averaged:

    JUDGE   faithfulness 4.6/5   relevancy 4.4/5   clarity 4.8/5

WHAT IT MEANS, IN PLAIN WORDS
-----------------------------
    faithfulness 4.6   the explanations mostly stick to what the data shows.
                       A drop here is the one to take seriously: it usually
                       means an answer claimed a CAUSE, e.g. "flows fell
                       BECAUSE of the downgrade", when rows can only show that
                       two things happened in the same month.
    relevancy 4.4      the answers address the question asked. A low score
                       often means it answered a nearby question instead.
    clarity 4.8        a business reader would follow it without help.

Below about 4.0 on faithfulness is worth investigating. Small movements,
0.1 or 0.2, mean nothing: this is an opinion, and opinions wobble.

WHAT THE JUDGE IS NOT SHOWN
---------------------------
The judge sees the question, the first rows of the result, and the
explanation. It is NOT given the expected answer, so it cannot simply agree
with the answer key.
"""

from metadata import llm

JUDGE_PROMPT = """You are grading an explanation written for a business user about a mutual fund database.

You will be given the question, the rows of data the explanation was based on, and the explanation.

Score each of these from 1 to 5, where 5 is best:

  faithfulness: does the explanation say ONLY what the rows support? Score 1 if it states
    a figure or a fact the rows do not show. Score 1 if it claims one thing CAUSED another,
    since rows can show timing but never cause.
  relevancy: does it answer the question that was asked, rather than a related one?
  clarity: would a business person with no database knowledge understand it?

Reply with a JSON object only:
{"faithfulness": 1-5, "relevancy": 1-5, "clarity": 1-5, "reason": "<one short sentence>"}"""


def _rows_as_text(result: dict, limit: int = 20) -> str:
    """The first rows of the result, small enough to fit in a prompt."""
    result = result or {}
    columns = result.get("columns") or []
    rows = (result.get("rows") or [])[:limit]
    lines = [" | ".join(str(c) for c in columns)]
    lines += [" | ".join("" if v is None else str(v) for v in row) for row in rows]
    return "\n".join(lines) if rows else "(no rows)"


def judge(question: str, result: dict, commentary: str) -> dict:
    """
    Ask the chat model to score one explanation.

    Returns {"faithfulness": n, "relevancy": n, "clarity": n, "reason": str},
    or {"error": "..."} if the model could not be reached or replied oddly.
    The evaluation carries on either way: a judge failure must never stop a
    run that has already produced real numbers.
    """
    if not commentary:
        return {"faithfulness": 0, "relevancy": 0, "clarity": 0, "reason": "No explanation was written."}

    user = (f"QUESTION:\n{question}\n\n"
            f"ROWS THE EXPLANATION WAS BASED ON:\n{_rows_as_text(result)}\n\n"
            f"EXPLANATION:\n{commentary}")
    try:
        answer = llm.chat_json("strong", JUDGE_PROMPT, user)
    except Exception as error:
        return {"error": str(error)[:200]}

    def as_score(value) -> int:
        """Keep whatever the model returns inside 1 to 5, or 0 if unusable."""
        try:
            return max(1, min(5, int(float(value))))
        except (TypeError, ValueError):
            return 0

    return {
        "faithfulness": as_score(answer.get("faithfulness")),
        "relevancy": as_score(answer.get("relevancy")),
        "clarity": as_score(answer.get("clarity")),
        "reason": str(answer.get("reason", ""))[:300],
    }