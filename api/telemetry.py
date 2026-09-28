"""
api/telemetry.py  (AWS version)
===============================

WHAT THIS FILE IS FOR
---------------------
Sends traces of every request, and of every agent step, to AWS X-Ray, where
they appear in the CloudWatch console. Two calls, in api/main.py:

    telemetry.setup()          # before the FastAPI app is created
    telemetry.instrument(app)  # right after it is created

HOW THE TRACES GET TO AWS
-------------------------
    this app ──OTLP──▶ AWS OpenTelemetry collector ──▶ AWS X-Ray ──▶ CloudWatch console
                      (Docker container "otel-collector",
                       listening on localhost:4318)

OpenTelemetry is the standard way to send traces. The app sends them to the
collector over OTLP (OpenTelemetry's own format). The collector signs them
with your AWS credentials and forwards them to X-Ray. That keeps AWS
credentials and AWS-specific code out of the app itself: to send traces
somewhere else, only the collector's config changes.

The collector is started by compose.yaml, with its settings in
docker/otel/collector.yaml.

WHAT IS RECORDED
----------------
    - every HTTP request to FastAPI (FastAPIInstrumentor)
    - every span the agent opens: agent.ask, agent.write_sql, ...
      (see agent/graph.py), with their attributes

WHERE TO SEE IT
---------------
CloudWatch console -> X-Ray traces -> Traces (and Trace Map). Data appears
within about a minute.

If OTEL_EXPORTER_OTLP_ENDPOINT is not set in .env, nothing is sent and the
spans cost nothing: the app runs exactly the same, just without tracing.
"""

import os

from dotenv import load_dotenv


def setup() -> bool:
    load_dotenv()
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        print("Tracing: OTEL_EXPORTER_OTLP_ENDPOINT not set, skipping")
        return False

    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.extension.aws.trace import AwsXRayIdGenerator
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(
        # service.name is how this app is labelled in X-Ray and CloudWatch.
        resource=Resource.create({"service.name": os.environ.get("OTEL_SERVICE_NAME", "fundgraph-api")}),
        # X-Ray trace IDs start with the time the trace began. This generator
        # makes IDs in that form, so X-Ray files every trace correctly.
        id_generator=AwsXRayIdGenerator(),
    )
    # BatchSpanProcessor collects finished spans and sends them in groups in
    # the background, so tracing never slows down answering a question.
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint.rstrip("/") + "/v1/traces")))
    trace.set_tracer_provider(provider)
    print(f"Tracing: sending traces to {endpoint} (AWS X-Ray via the OpenTelemetry collector)")
    return True


def instrument(app) -> None:
    """Record every HTTP request to the FastAPI app as a trace."""
    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    FastAPIInstrumentor.instrument_app(app)