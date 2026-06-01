import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

logger = logging.getLogger("ai_service.observability.tracing")

_in_memory_exporter = None

def setup_tracing(service_name: str = "ai-service", environment: str = "development") -> TracerProvider:
    """
    Initializes the OpenTelemetry TracerProvider and registers standard processors.
    Safe to call multiple times; retrieves the active provider if already configured.
    """
    global _in_memory_exporter

    from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
    provider = trace.get_tracer_provider()

    if not isinstance(provider, SDKTracerProvider):
        resource = Resource.create(attributes={
            "service.name": service_name,
            "service.environment": environment
        })
        provider = SDKTracerProvider(resource=resource)
        trace.set_tracer_provider(provider)

    if environment in ("test", "development"):
        if _in_memory_exporter is None:
            try:
                from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
                _in_memory_exporter = InMemorySpanExporter()
                provider.add_span_processor(SimpleSpanProcessor(_in_memory_exporter))
                logger.info("Initialized InMemorySpanExporter for tracing.")
            except ImportError as e:
                logger.warning(f"InMemorySpanExporter not found; tracing will run without exporter: {e}")
    else:
        try:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
            logger.info("Initialized ConsoleSpanExporter for tracing.")
        except ImportError:
            logger.warning("ConsoleSpanExporter not found; tracing will run without exporter.")

    return provider

def get_tracer(name: str = "ai_service"):
    """Returns a tracer instance."""
    return trace.get_tracer(name)

def get_in_memory_exporter():
    """Returns the globally configured InMemorySpanExporter (for unit testing/debugging)."""
    global _in_memory_exporter
    return _in_memory_exporter

def instrument_app(app):
    """Instruments a FastAPI application with OpenTelemetry."""
    FastAPIInstrumentor.instrument_app(app)
    logger.info("FastAPI application instrumented with OpenTelemetry.")
