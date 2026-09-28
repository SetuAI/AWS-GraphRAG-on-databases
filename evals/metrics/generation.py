"""
evals/metrics/generation.py
===========================

WHY WE DO THIS
--------------
The agent can find the right tables, write correct SQL, and still produce a
bad answer: an explanation that quotes a number which is not in the result,
that answers a different question, or that answers at all when it should have
refused. The first of those is the most dangerous thing a system like this can
do, because the table beside it looks authoritative.

So the explanation is checked against the rows it was supposedly based on.

WHAT WE ARE EVALUATING
----------------------
The ANSWERING part: steps 9 to 11, plus the refusals.

    groundedness    every number in the explanation appears in the result rows
    completeness    the explanation mentions what the question was about
    refusal         a question that had to be refused was refused
    sql_valid       exactly one SELECT was produced and every check passed
    executed        the query ran and returned rows

WHY GROUNDEDNESS IS CHECKED WITH ARITHMETIC, NOT A MODEL
--------------------------------------------------------
The failure that matters here is an invented number: the explanation says
"flows fell by 268 lakh" when no such figure is in the result. That is
catchable exactly, by pulling every number out of the text and looking for
it in the rows.

A model judge would give an opinion about that, with a different opinion
tomorrow. This check gives the same answer every time, costs nothing, and
names the number that was not found. The model judge (evals/metrics/judge.py)
is for the parts arithmetic cannot see, such as whether the wording is
misleading.

WHAT COUNTS AS A MATCH
----------------------
A number in the text counts as grounded if it appears in the rows, allowing
for rounding and formatting:

    text "268.4"      rows 268.4132      -> grounded (rounded)
    text "2,68,412"   rows 268412        -> grounded (separators ignored)
    text "24%"        rows 24.0          -> grounded
    text "2024"       any year in a date -> grounded (dates are not figures)

Small whole numbers (0, 1, 2, ... 12) are ignored, because they appear in
ordinary sentences ("the top 5 schemes", "three of them") rather than as
claims about the data.

WHAT THE EXPECTED OUTPUT LOOKS LIKE
-----------------------------------
For one question:

    status_correct true
    groundedness   {"checked": 4, "grounded": 4, "score": 1.0, "ungrounded": []}
    completeness   {"checked": 2, "found": 2, "score": 1.0, "missing": []}
    sql            {"sql_written": true, "checks_passed": true,
                    "executed": true, "attempts": 1}

WHAT IT MEANS, IN PLAIN WORDS
-----------------------------
    status_correct   the question ended the way it should have: answered when
                     it should be answered, refused when it should be refused.
    groundedness 1.0 the explanation quoted 4 figures and all 4 appear in the
                     rows. Nothing was invented. A score of 0.75 with
                     "ungrounded": [268.4] means one figure was made up, and
                     tells you exactly which.
    completeness 1.0 the explanation mentioned both things the question was
                     about, e.g. the fund name and the year. "missing":
                     ["konkan"] means it answered without ever naming the
                     distributor, which reads as evasive even if the rows are
                     right.
    sql attempts 1   the query was accepted first time. 2 or 3 means it failed
                     a safety check or the database, and was rewritten. Rising
                     retries are an early warning that the model is struggling.
"""

import re

# Numbers at or below this are treated as ordinary words, not data claims.
SMALL_NUMBER_LIMIT = 12

# How close a number in the text must be to one in the rows, as a share of
# the row value. 0.01 allows the rounding that any readable sentence needs.
TOLERANCE = 0.01


def _numbers_in_text(text: str) -> list[float]:
    """
    Every number in a piece of text, as plain numbers.

    Handles "1,234.5", "2,68,412" (Indian grouping), "24%", "-15.2".
    """
    found = []
    for raw in re.findall(r"-?\d[\d,]*\.?\d*", text or ""):
        cleaned = raw.replace(",", "").rstrip(".")
        try:
            found.append(float(cleaned))
        except ValueError:
            continue
    return found


def _numbers_in_rows(result: dict) -> list[float]:
    """
    Every number the explanation is allowed to quote.

    That is more than the numbers printed in the rows, because a good
    explanation does arithmetic a reader would otherwise have to do:

      1. the values in the rows themselves
      2. the parts of any date, so "March 2024" can be matched
      3. the TOTAL of each numeric column, since "40 schemes in total" is a
         fair summary of rows holding 8, 7, 12, 13
      4. the NUMBER OF ROWS, since "five schemes are above the limit" is a
         fair summary of a five-row result

    Without 3 and 4, every sentence that adds up a column would be reported
    as invented, which would make the whole measure useless.
    """
    rows = (result or {}).get("rows", []) or []
    found: list[float] = []
    columns: dict[int, list[float]] = {}

    for row in rows:
        for position, value in enumerate(row):
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                found.append(float(value))
                columns.setdefault(position, []).append(float(value))
            elif value is not None:
                text = str(value)
                # Dates arrive as text, e.g. "2024-03-01". Their parts count.
                found.extend(float(part) for part in re.findall(r"\d+", text)[:3])
                for number in _numbers_in_text(text):
                    found.append(number)
                    columns.setdefault(position, []).append(number)

    for values in columns.values():
        found.append(sum(values))          # the column total
    found.append(float(len(rows)))         # how many rows came back
    return found


def _is_grounded(number: float, allowed: list[float]) -> bool:
    """
    Is this number in the rows, allowing for rounding, units and sign?

    THE SIGN IS IGNORED on purpose. A row holding -268413.2 is written in an
    explanation as "flows FELL BY 268.4 lakh": the direction is carried by the
    words, not by a minus sign, so insisting on the sign would mark good
    explanations as invented.
    """
    claimed = abs(number)
    for value in allowed:
        actual = abs(value)
        if actual == claimed:
            return True
        # Rounding: 268413.2 written as 268413 or 268413.2
        scale = max(actual, 1.0)
        if abs(actual - claimed) <= TOLERANCE * scale:
            return True
        if round(actual, 1) == claimed or round(actual) == claimed:
            return True
        # Units: the same figure written in hundreds, thousands, lakh or crore.
        for divisor in (100, 1000, 100000, 10000000):
            scaled = actual / divisor
            if abs(scaled - claimed) <= TOLERANCE * max(scaled, 1.0):
                return True
            if round(scaled, 1) == claimed or round(scaled) == claimed:
                return True
    return False


def _plausible_year(number: float) -> bool:
    """Is this a year rather than a figure? 1990 to 2100, whole."""
    return float(number).is_integer() and 1990 <= number <= 2100


def groundedness(commentary: str, result: dict, question: str = "", sql: str = "") -> dict:
    """
    Every number in the explanation, checked against the result rows.

    Returns {"checked": n, "grounded": n, "score": 0..1, "ungrounded": [...]}.
    A score of 1.0 means the explanation invented nothing.

    THREE KINDS OF NUMBER ARE NOT CLAIMS ABOUT THE DATA, AND ARE SKIPPED
    -------------------------------------------------------------------
    1. Numbers from the QUESTION. Asked "downgraded during 2024?", a good
       answer says "during 2024" even when the rows hold no dates.
    2. Years. "at the last month end of July 2026" describes when, not how
       much, and the period usually comes from the query rather than a row.
    3. Numbers from the SQL itself, such as a threshold or a limit:
       "considering distributors with at least 100 transactions" repeats the
       HAVING clause. That is the explanation being transparent about the
       calculation, which the prompt asks it to do.

    Everything else must be traceable to the rows, their column totals, or
    the number of rows. UNIT CONVERSIONS ARE NOT EXCUSED: a figure written in
    crore must be the row value divided by 10,000,000, or it is wrong.
    """
    skip = {abs(n) for n in _numbers_in_text(question)} | {abs(n) for n in _numbers_in_text(sql)}
    claimed = [n for n in _numbers_in_text(commentary)
               if abs(n) > SMALL_NUMBER_LIMIT
               and abs(n) not in skip
               and not _plausible_year(abs(n))]
    if not claimed:
        # No figures quoted: nothing can be invented, so this passes.
        return {"checked": 0, "grounded": 0, "score": 1.0, "ungrounded": []}

    allowed = _numbers_in_rows(result)
    ungrounded = [n for n in claimed if not _is_grounded(n, allowed)]
    return {
        "checked": len(claimed),
        "grounded": len(claimed) - len(ungrounded),
        "score": round((len(claimed) - len(ungrounded)) / len(claimed), 3),
        "ungrounded": ungrounded,
    }


def completeness(commentary: str, must_mention: list[str]) -> dict:
    """
    Does the explanation mention what the question was about?

    must_mention holds the entity, period or metric the question named, e.g.
    ["konkan"] or ["credit risk", "2024"]. An answer that returns the right
    rows but never names the fund is a poor answer, and this catches it.
    """
    if not must_mention:
        return {"checked": 0, "found": 0, "score": 1.0, "missing": []}
    text = (commentary or "").lower()
    missing = [word for word in must_mention if word.lower() not in text]
    return {
        "checked": len(must_mention),
        "found": len(must_mention) - len(missing),
        "score": round((len(must_mention) - len(missing)) / len(must_mention), 3),
        "missing": missing,
    }


def status_correct(result: dict, expected_status: str) -> bool:
    """
    Did the question end the way it should?

    For a refusal question, "blocked" is the right answer and "completed" is
    a failure, however good the explanation.
    """
    return result.get("status") == expected_status


def sql_quality(result: dict) -> dict:
    """
    Was a usable query produced, and did it run?

        sql_written  the model produced SQL at all
        checks_passed  every safety check passed (SELECT only, known tables,
                       a real date range, cost within the limit)
        executed     the query ran and rows came back
        attempts     how many tries it took; more than 1 means it was rewritten
    """
    checks = result.get("checks") or []
    return {
        "sql_written": bool(result.get("sql")),
        "checks_passed": bool(checks) and all(c.get("passed") for c in checks),
        "executed": bool((result.get("result") or {}).get("rows")),
        "attempts": result.get("attempts", 0),
    }


def score(result: dict, expected: dict) -> dict:
    """Every generation number for one question."""
    commentary = result.get("commentary") or ""
    return {
        "status_correct": status_correct(result, expected.get("expect_status", "completed")),
        "status": result.get("status"),
        "groundedness": groundedness(commentary, result.get("result") or {},
                                      expected.get("question", ""), result.get("sql") or ""),
        "completeness": completeness(commentary, expected.get("must_mention") or []),
        "sql": sql_quality(result),
    }