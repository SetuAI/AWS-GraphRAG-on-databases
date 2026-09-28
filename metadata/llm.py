"""
metadata/llm.py  (AWS version)
==============================

WHAT THIS FILE IS FOR
---------------------
The ONE place in the project that calls Amazon Bedrock. Every other file
calls one of these functions and never talks to Bedrock itself:

    chat_json(role, system, user) -> ask the chat model, get back a Python dict
    embed(texts)                  -> turn texts into lists of 1,024 numbers
    bedrock_client()              -> the shared connection (the guardrail check uses it too)

WHERE THE SETTINGS COME FROM (.env)
-----------------------------------
    AWS_ACCESS_KEY        the IAM user's access key
    AWS_SECRET_KEY        its secret key
    AWS_MODEL_BEDROCK_ID  the chat model to call, for every role
    AWS_BEDROCK_ARN       that model's ARN; the REGION is read from it:
                          arn:aws:bedrock:us-east-1:123456789012:...
                                          ^^^^^^^^^ region
    AWS_REGION            (only checked here) must match the ARN's region, because
                          the tracing container uses it; a mismatch prints a warning

The embedding model is set in config/models.yaml (Titan Text Embeddings V2)
and is called with the same keys, in the same region.

HOW IT CALLS BEDROCK
--------------------
    chat       -> bedrock-runtime Converse API: one request format for every
                  chat model on Bedrock
    embeddings -> bedrock-runtime InvokeModel with Titan's own request format

HOW WE GET JSON BACK
--------------------
Converse has no "reply with JSON only" switch. Every system prompt in this
project asks for a JSON object, and _extract_json() takes the text from the
first "{" to the last "}" and parses that, so a reply wrapped in a sentence
or ``` fences still works.

LANGSMITH
---------
@traceable records each call in LangSmith (prompt, reply, time) when
LANGSMITH_TRACING=true in .env. Otherwise it does nothing.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import boto3
import yaml
from botocore.config import Config
from dotenv import load_dotenv
from langsmith import traceable

MODELS_FILE = Path(__file__).resolve().parents[1] / "config" / "models.yaml"

# The .env settings this file cannot work without.
REQUIRED = ["AWS_ACCESS_KEY", "AWS_SECRET_KEY", "AWS_MODEL_BEDROCK_ID", "AWS_BEDROCK_ARN"]


@lru_cache(maxsize=1)
def _models() -> dict:
    """Read config/models.yaml once, then reuse it."""
    return yaml.safe_load(MODELS_FILE.read_text())


def region_from_arn(arn: str) -> str:
    """
    The region part of an ARN. ARNs are colon-separated:
        arn : aws : bedrock : us-east-1 : 123456789012 : inference-profile/...
         0     1      2         3            4             5
    """
    parts = arn.split(":")
    if len(parts) < 6 or parts[0] != "arn" or not parts[3]:
        raise ValueError(f"AWS_BEDROCK_ARN does not look like a Bedrock ARN with a region: {arn!r}")
    return parts[3]


@lru_cache(maxsize=1)
def aws_settings() -> dict:
    """
    Read and check the AWS settings from .env, once.
    Stops with a clear message naming any missing setting, instead of failing
    later with a less obvious AWS error.
    """
    load_dotenv()
    missing = [name for name in REQUIRED if not os.environ.get(name, "").strip()]
    if missing:
        raise RuntimeError(f"Missing in .env: {', '.join(missing)} (see .env.example)")

    region = region_from_arn(os.environ["AWS_BEDROCK_ARN"].strip())
    tracing_region = os.environ.get("AWS_REGION", "").strip()
    if tracing_region and tracing_region != region:
        print(f"WARNING: AWS_REGION={tracing_region} but AWS_BEDROCK_ARN is in {region}. "
              f"Bedrock calls use {region}; set AWS_REGION={region} so traces go to the same region.")

    return {
        "access_key": os.environ["AWS_ACCESS_KEY"].strip(),
        "secret_key": os.environ["AWS_SECRET_KEY"].strip(),
        "model_id": os.environ["AWS_MODEL_BEDROCK_ID"].strip(),
        "region": region,
    }


@lru_cache(maxsize=1)
def bedrock_client():
    """
    Create the Bedrock runtime connection once, then reuse it.

    The keys are passed in directly from .env, so no AWS CLI setup or
    ~/.aws folder is needed on the machine.

    retries "adaptive" with max_attempts 8: if Bedrock answers
    "ThrottlingException" (too many requests), boto3 waits and tries again,
    slowing down automatically, instead of failing straight away.
    """
    s = aws_settings()
    return boto3.client(
        "bedrock-runtime",
        region_name=s["region"],
        aws_access_key_id=s["access_key"],
        aws_secret_access_key=s["secret_key"],
        config=Config(retries={"mode": "adaptive", "max_attempts": 8}, read_timeout=120),
    )


def model_for(role: str) -> str:
    """
    Which model a role uses. Both "small" and "strong" use AWS_MODEL_BEDROCK_ID
    today; the role is kept so a second model can be added here later.
    """
    if role not in ("small", "strong"):
        raise ValueError(f"Unknown model role: {role!r}")
    return aws_settings()["model_id"]


def _extract_json(text: str) -> dict:
    """Parse the JSON object inside a model reply, ignoring anything around it."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in model reply: {text[:200]}")
    return json.loads(text[start:end + 1])


@traceable(run_type="llm", name="bedrock.chat_json")
def chat_json(role: str, system: str, user: str) -> dict:
    """
    Ask the chat model a question and get its answer as a Python dict.
    temperature 0 makes replies as consistent as possible: the same question
    should produce the same SQL each time.
    """
    reply = bedrock_client().converse(
        modelId=model_for(role),
        system=[{"text": system + "\n\nReply with the JSON object only, no other text."}],
        messages=[{"role": "user", "content": [{"text": user}]}],
        inferenceConfig={"maxTokens": _models()["max_tokens"], "temperature": 0},
    )
    # The reply is a list of content blocks; a normal text reply has one.
    text = "".join(block.get("text", "") for block in reply["output"]["message"]["content"])
    return _extract_json(text)


def _embed_one(text: str) -> list[float]:
    """
    Titan Text Embeddings V2 takes ONE text per request.
    normalize=true scales each vector to length 1, the right form for
    cosine-distance search in pgvector.
    """
    models = _models()
    response = bedrock_client().invoke_model(
        modelId=models["embedding"],
        body=json.dumps({"inputText": text, "dimensions": models["embedding_dimensions"], "normalize": True}),
    )
    return json.loads(response["body"].read())["embedding"]


@traceable(run_type="embedding", name="bedrock.embed")
def embed(texts: list[str]) -> list[list[float]]:
    """
    Turn each text into its embedding. Titan handles one text per request,
    so several requests run at the same time (8 threads) to keep this fast.
    pool.map returns results in the same order as the texts went in.
    """
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(_embed_one, texts))