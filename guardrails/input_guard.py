"""
guardrails/input_guard.py  (AWS version)
========================================

WHAT THIS FILE IS FOR
---------------------
Checks a user's question with an Amazon Bedrock Guardrail BEFORE anything
else happens. The guardrail has one filter switched on: PROMPT_ATTACK, which
detects attempts to override the AI's instructions ("ignore your rules",
"you are now in developer mode", ...). A flagged question is stopped: no
model sees it, no SQL is written.

WHERE THE SETTINGS COME FROM (.env)
-----------------------------------
    BEDROCK_GUARDRAIL_ID   the guardrail to use
    BEDROCK_VERSION        its published version, e.g. 1
The keys and region are the same as for the models (see metadata/llm.py):
the guardrail must be in the region of AWS_BEDROCK_ARN.

HOW IT IS CALLED
----------------
bedrock-runtime ApplyGuardrail: checks a piece of text against a guardrail
without calling any model. source="INPUT" means "this is what a user sent".
The reply's "action" is:
    GUARDRAIL_INTERVENED -> the guardrail flagged it
    NONE                 -> nothing flagged

WHAT IT DOES NOT CATCH
----------------------
A plain request to change data ("delete the transactions table") is not an
instruction override, so the guardrail may let it through. That is caught
later, twice: the understanding step refuses data-change requests, and the
SQL check allows SELECT statements only.
"""

import os

from dotenv import load_dotenv

from metadata.llm import bedrock_client


def check(question: str, fail_open: bool) -> dict:
    """
    Returns {"attack": bool, "checked": bool, "note": str}.
    checked is False when the guardrail could not be reached; then attack
    follows fail_open (see config/agent.yaml).
    """
    load_dotenv()
    guardrail_id = os.environ.get("BEDROCK_GUARDRAIL_ID", "").strip()
    version = os.environ.get("BEDROCK_VERSION", "").strip() or "1"
    if not guardrail_id:
        return {"attack": not fail_open, "checked": False, "note": "BEDROCK_GUARDRAIL_ID not set in .env"}

    try:
        result = bedrock_client().apply_guardrail(
            guardrailIdentifier=guardrail_id,
            guardrailVersion=version,
            source="INPUT",
            content=[{"text": {"text": question}}],
        )
    except Exception as error:   # network failure, missing permission, wrong ID or region, ...
        return {"attack": not fail_open, "checked": False, "note": f"Check failed: {error}"}

    attack = result.get("action") == "GUARDRAIL_INTERVENED"
    return {"attack": attack, "checked": True,
            "note": "Prompt attack detected by Bedrock Guardrail" if attack else "No prompt attack detected"}