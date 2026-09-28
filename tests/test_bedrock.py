"""
tests/test_bedrock.py
=====================

WHAT THIS FILE IS FOR
---------------------
Checks that the code can reach Amazon Bedrock with the keys in .env, and that
the models answer:

    AWS_MODEL_BEDROCK_ID  (the chat model, both roles)  -> returns JSON
    Titan Text Embeddings V2                            -> returns 1,024 numbers

Run this FIRST after filling in .env. If it passes, the keys, the region in
AWS_BEDROCK_ARN, the model ID and the IAM permissions are all correct.

    pytest tests/test_bedrock.py -v

COMMON FAILURES
---------------
    Missing in .env: ...                    -> fill in that setting
    UnrecognizedClientException             -> AWS_ACCESS_KEY / AWS_SECRET_KEY are wrong
    AccessDeniedException ... use case      -> submit the Anthropic form (instructions.pdf)
    AccessDeniedException ... marketplace   -> the IAM policy's aws-marketplace permissions
    ValidationException ... on-demand       -> AWS_MODEL_BEDROCK_ID must be the inference
                                               profile ID (us.anthropic....), not the plain model ID
"""

import os

import pytest
from dotenv import load_dotenv

from metadata import llm

load_dotenv()


def test_env_settings_are_filled_in():
    settings = llm.aws_settings()
    assert settings["access_key"] and settings["secret_key"] and settings["model_id"]
    assert settings["region"], "region could not be read from AWS_BEDROCK_ARN"


def test_tracing_region_matches_model_region():
    """The X-Ray collector uses AWS_REGION; it must be the region of the model's ARN."""
    assert os.environ.get("AWS_REGION", "").strip() == llm.aws_settings()["region"]


@pytest.mark.parametrize("role", ["small", "strong"])
def test_chat_model_returns_json(role):
    answer = llm.chat_json(role, 'Reply with a JSON object: {"status": "OK"}', "Say OK.")
    assert answer.get("status") == "OK"


def test_embedding_has_1024_dimensions():
    vectors = llm.embed(["Sahyadri Credit Risk Fund", "Konkan Wealth Partners"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 1024