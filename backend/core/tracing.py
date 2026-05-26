from __future__ import annotations

import logging
import os
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from backend.core.config import QDRANT_URL
import sentry_sdk
from backend.core.config import JWT_SECRET_KEY

logger = logging.getLogger(__name__)


def init_tracing(app):
    otlp_endpoint = os.getenv("DOCUMIND_OTLP_ENDPOINT")
    # Initialize Sentry if configured (independent of OTLP)
    sentry_dsn = os.getenv("DOCUMIND_SENTRY_DSN")
    if sentry_dsn:
        try:
            sentry_sdk.init(dsn=sentry_dsn)
            logger.info("Sentry initialized")
        except Exception as exc:
            logger.warning("Failed to init Sentry: %s", exc)

    if not otlp_endpoint:
        logger.info("OTLP endpoint not configured; tracing disabled")
        return

    resource = Resource.create({"service.name": "documind-backend"})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    # instrument FastAPI app
    try:
        FastAPIInstrumentor.instrument_app(app)
    except Exception as exc:
        logger.warning("Failed to instrument FastAPI for OpenTelemetry: %s", exc)
