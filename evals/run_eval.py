"""
evals/run_eval.py
=================

WHY WE DO THIS
--------------
Without measurement, "the system is working well" is an opinion. Every time
you change a prompt, switch the model in .env, or adjust how tables are
found, something gets better and something else may quietly get worse, and
trying three questions by hand will not tell you which.

This file asks the SAME 18 questions every time and scores the answers the
SAME way, so two runs can be compared honestly.

WHAT WE ARE EVALUATING
----------------------
Three separate things, kept apart on purpose, because a failure in each has
a different cause and a different fix:

    1. FINDING     did it pick the right tables and join them correctly?
                   (a problem here means the notes, embeddings or graph)
    2. ANSWERING   is the explanation supported by the rows, and does it
                   answer the question?  (a problem here means the model or
                   the prompt)
    3. REFUSING    were the questions that must be refused actually refused?
                   (a problem here is a safety problem, not a quality one)

HOW TO RUN IT
-------------
    python -m evals.run_eval                  the deterministic numbers
    python -m evals.run_eval --judge          plus a model's opinion
    python -m evals.run_eval --ragas          plus RAGAS
    python -m evals.run_eval --only p5,h1     just those questions, while fixing one

It takes 3 to 8 minutes and makes about 70 Bedrock calls, so it costs a few
rupees per run. Run it after a change, not after every save.

WHAT THE EXPECTED OUTPUT LOOKS LIKE
-----------------------------------
    18 questions · apac.amazon.nova-pro-v1:0 · 4m 12s

    FINDING     precision@5 0.78   recall@5 0.94   MRR 0.91   join path 15/18
    ANSWERING   grounded 17/18     complete 16/18  SQL ran 17/18
    REFUSING    3/3
    OVERALL     status correct 17/18

WHAT EACH NUMBER MEANS, IN PLAIN WORDS
--------------------------------------
    precision@5 0.78   of the tables it chose, about 3 in 4 were needed. The
                       rest were extra. Spare tables are untidy, not fatal.
    recall@5 0.94      of the tables actually needed, it found nearly all. This
                       is the one to watch: a MISSING table means the question
                       cannot be answered at all.
    MRR 0.91           the first correct table was usually right at the top.
    join path 15/18    15 questions used every join they needed, including the
                       hidden links. The 3 that did not are named in the report.
    grounded 17/18     in 17 answers, every number in the explanation was found
                       in the actual rows. One answer quoted a figure that was
                       not there, which is the single worst failure a system
                       like this can have.
    complete 16/18     16 explanations mentioned the fund, period or metric the
                       question asked about.
    SQL ran 17/18      17 queries passed every safety check and returned rows.
    REFUSING 3/3       all three questions that had to be refused were refused.
    status correct     how many questions ended the way they should have:
                       answered when they should be, refused when they should be.

HOW TO READ A CHANGE BETWEEN RUNS
---------------------------------
With 18 questions, one question flipping moves a score by about 0.06. Treat
anything smaller than that as noise, not progress. What matters is a number
moving by SEVERAL questions, or a specific question that used to pass and
now does not, which the per-question file shows you.

WHERE RESULTS GO
----------------
    evals/results/<timestamp>.json    every question, with its scores
    evals/results/latest.json         a copy of the most recent run

Keep them: comparing today's run with last week's is the whole point.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

from agent.graph import ask
from evals.metrics import generation, retrieval

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_FILE = ROOT / "evals" / "questions" / "questions.yaml"
RESULTS_FOLDER = ROOT / "evals" / "results"


def load_questions(only: str | None = None) -> list[dict]:
    """Read the question set, optionally keeping only the ids listed in --only."""
    questions = yaml.safe_load(QUESTIONS_FILE.read_text())["questions"]
    if only:
        wanted = {part.strip() for part in only.split(",") if part.strip()}
        # A partial id is enough: --only p5 matches p5_rejection_rate.
        questions = [q for q in questions
                     if q["id"] in wanted or any(q["id"].startswith(w) for w in wanted)]
    return questions


def run_one(question: dict, use_judge: bool) -> dict:
    """
    Ask one question and score the answer.

    Nothing here stops the run: if the agent raises, the question is recorded
    as an error and the rest carry on, so one broken question never costs you
    the whole 4 minutes.
    """
    started = time.perf_counter()
    try:
        # Each question gets its own conversation id, so nothing carries over
        # from the question before it. The evaluation must test one question
        # at a time, not a conversation.
        result = ask(question["question"], conversation_id=f"eval-{question['id']}")
        error = None
    except Exception as failure:
        result, error = {}, str(failure)[:300]

    scored = {
        "id": question["id"],
        "kind": question["kind"],
        "question": question["question"],
        "seconds": round(time.perf_counter() - started, 1),
        "error": error,
        "retrieval": retrieval.score(result, question),
        "generation": generation.score(result, question),
        "result": {
            "status": result.get("status"),
            "block_reason": result.get("block_reason"),
            "error": result.get("error"),
            "checks": result.get("checks"),
            "attempts": result.get("attempts"),
            "tables": result.get("tables"),
            "sql": result.get("sql"),
            "commentary": result.get("commentary"),
            "result": result.get("result"),
            "join_path": result.get("join_path"),
        },
        "expected": question,
    }

    if use_judge and result.get("commentary"):
        from evals.metrics import judge as judge_module
        scored["judge"] = judge_module.judge(question["question"],
                                             result.get("result") or {},
                                             result.get("commentary"))
    return scored


def summarise(runs: list[dict]) -> dict:
    """Average the per-question scores into the headline numbers."""
    def mean(values):
        return round(sum(values) / len(values), 3) if values else 0.0

    answered = [r for r in runs if r["expected"].get("expect_status") == "completed"]
    refusals = [r for r in runs if r["expected"].get("expect_status") == "blocked"]

    summary = {
        "questions": len(runs),
        "finding": {
            "precision_at_k": mean([r["retrieval"]["precision_at_k"] for r in answered]),
            "recall_at_k": mean([r["retrieval"]["recall_at_k"] for r in answered]),
            "f1_at_k": mean([r["retrieval"]["f1_at_k"] for r in answered]),
            "mrr": mean([r["retrieval"]["reciprocal_rank"] for r in answered]),
            "join_path_exact": sum(1 for r in answered if r["retrieval"]["join_path"]["exact"]),
            "join_path_of": len(answered),
        },
        "answering": {
            "grounded": sum(1 for r in answered if r["generation"]["groundedness"]["score"] == 1.0),
            "grounded_of": len(answered),
            "complete": sum(1 for r in answered if r["generation"]["completeness"]["score"] == 1.0),
            "complete_of": len(answered),
            "sql_ran": sum(1 for r in answered if r["generation"]["sql"]["executed"]),
            "sql_of": len(answered),
            "retries": sum(max(0, r["generation"]["sql"]["attempts"] - 1) for r in answered),
        },
        "refusing": {
            "correct": sum(1 for r in refusals if r["generation"]["status_correct"]),
            "of": len(refusals),
        },
        "overall": {
            "status_correct": sum(1 for r in runs if r["generation"]["status_correct"]),
            "of": len(runs),
            "errors": sum(1 for r in runs if r["error"]),
        },
    }

    judged = [r["judge"] for r in runs if isinstance(r.get("judge"), dict) and "faithfulness" in r["judge"]]
    if judged:
        summary["judge"] = {
            "faithfulness": mean([j["faithfulness"] for j in judged]),
            "relevancy": mean([j["relevancy"] for j in judged]),
            "clarity": mean([j["clarity"] for j in judged]),
            "questions": len(judged),
        }
    return summary


def print_report(summary: dict, runs: list[dict], model: str, seconds: float) -> None:
    """The scorecard, plus a line per question that did not do what it should."""
    f, a, r, o = summary["finding"], summary["answering"], summary["refusing"], summary["overall"]
    minutes = f"{int(seconds // 60)}m {int(seconds % 60)}s"

    print()
    print(f"{summary['questions']} questions · {model} · {minutes}")
    print("=" * 74)
    print(f"FINDING     precision@5 {f['precision_at_k']:.2f}   recall@5 {f['recall_at_k']:.2f}   "
          f"MRR {f['mrr']:.2f}   join path {f['join_path_exact']}/{f['join_path_of']}")
    print(f"ANSWERING   grounded {a['grounded']}/{a['grounded_of']}     "
          f"complete {a['complete']}/{a['complete_of']}     "
          f"SQL ran {a['sql_ran']}/{a['sql_of']}   (SQL retries: {a['retries']})")
    print(f"REFUSING    {r['correct']}/{r['of']}")
    print(f"OVERALL     status correct {o['status_correct']}/{o['of']}"
          + (f"   errors {o['errors']}" if o["errors"] else ""))
    if "judge" in summary:
        j = summary["judge"]
        print(f"JUDGE       faithfulness {j['faithfulness']:.1f}/5   relevancy {j['relevancy']:.1f}/5   "
              f"clarity {j['clarity']:.1f}/5   (a model's opinion, not a measurement)")
    if "ragas" in summary:
        rg = summary["ragas"]
        print("RAGAS       " + (rg["skipped"] if "skipped" in rg else
                                f"faithfulness {rg['faithfulness']:.2f}   answer_relevancy {rg['answer_relevancy']:.2f}"))

    # Only the problems are listed: a clean run should be quiet.
    problems = []
    for run in runs:
        issues = []
        if run["error"]:
            issues.append(f"error: {run['error'][:60]}")
        if not run["generation"]["status_correct"]:
            issues.append(f"ended {run['result']['status']}, expected {run['expected']['expect_status']}")
            # For a failure, say WHY, so the next step is obvious: which
            # safety check refused the SQL, or what the database said.
            failed_checks = [c for c in (run["result"].get("checks") or []) if not c.get("passed")]
            if failed_checks:
                issues.append(f"check failed: {failed_checks[0].get('name')} - {str(failed_checks[0].get('detail'))[:80]}")
            elif run["result"].get("block_reason") or run["result"].get("error"):
                issues.append(str(run["result"].get("block_reason") or run["result"].get("error"))[:100])
        missing = run["retrieval"]["join_path"]["missing"]
        if missing:
            issues.append(f"missing join {missing[0]}")
        ungrounded = run["generation"]["groundedness"]["ungrounded"]
        if ungrounded:
            issues.append(f"figure not in the rows: {ungrounded[0]}")
        if run["generation"]["completeness"]["missing"]:
            issues.append(f"never mentions {run['generation']['completeness']['missing'][0]!r}")
        if issues:
            problems.append(f"  {run['id']:<24} {'; '.join(issues)}")

    if problems:
        print("-" * 74)
        print("Worth looking at:")
        print("\n".join(problems))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the agent against the question set.")
    parser.add_argument("--judge", action="store_true", help="also ask the model to score each explanation")
    parser.add_argument("--ragas", action="store_true", help="also score with RAGAS (needs: pip install ragas langchain-aws)")
    parser.add_argument("--only", help="run only these question ids, comma separated, e.g. p5,h1")
    parser.add_argument("--k", type=int, default=5, help="the k in precision@k and recall@k (default 5)")
    args = parser.parse_args()

    load_dotenv()
    import os
    model = os.environ.get("AWS_MODEL_BEDROCK_ID", "(model not set)")

    questions = load_questions(args.only)
    print(f"Asking {len(questions)} questions with {model} ...")

    started = time.perf_counter()
    runs = []
    for number, question in enumerate(questions, start=1):
        run = run_one(question, args.judge)
        runs.append(run)
        mark = "ok  " if run["generation"]["status_correct"] and not run["error"] else "LOOK"
        print(f"  {number:>2}/{len(questions)}  {mark}  {run['id']:<24} {run['seconds']:>5.1f}s")
    seconds = time.perf_counter() - started

    summary = summarise(runs)
    if args.ragas:
        from evals.metrics import ragas_adapter
        summary["ragas"] = ragas_adapter.evaluate(runs)

    print_report(summary, runs, model, seconds)

    # Save everything, so runs can be compared later.
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    report = {"when": stamp, "model": model, "seconds": round(seconds, 1),
              "summary": summary, "questions": runs}
    (RESULTS_FOLDER / f"{stamp}.json").write_text(json.dumps(report, indent=2, default=str))
    (RESULTS_FOLDER / "latest.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"Saved: evals/results/{stamp}.json")


if __name__ == "__main__":
    main()