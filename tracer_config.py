"""
Tracer configuration for Phoenix/OpenInference instrumentation.
This file isolates the tracer setup to avoid circular dependencies.
"""
import os
import json
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
import openinference.semconv.trace
import json
from neo4j.exceptions import CypherSyntaxError, Neo4jError
from dotenv import load_dotenv
load_dotenv()

# Load environment variables
ENV = os.getenv("ENV", "dev")
PHOENIX_API_KEY = os.getenv("PHOENIX_API_KEY")
print("PHOENIX_API_KEY: ",PHOENIX_API_KEY[:5]+"..")
DISABLE_TRACING = os.getenv("DISABLE_TRACING", "").lower() in {"1", "true", "yes", "on"}
PHOENIX_HOSTNAME = os.getenv("PHOENIX_HOSTNAME", "https://app.phoenix.arize.com/s/Palete_production")

# CRITICAL: Set OTEL headers in environment BEFORE importing phoenix.otel.register
# The register() function reads this environment variable internally
if PHOENIX_API_KEY and not os.getenv("OTEL_EXPORTER_OTLP_HEADERS"):
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"authorization=Bearer {PHOENIX_API_KEY}"
    print(f"✓ Set OTEL_EXPORTER_OTLP_HEADERS for authentication")
# Custom JSON encoder for Neo4j errors
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (CypherSyntaxError, Neo4jError)):
            return {
                "type": obj.__class__.__name__,
                "message": str(obj)
            }
        return super().default(obj)

# Set up custom JSON encoder
openinference.semconv.trace.JSON_ENCODER = CustomJSONEncoder()

# Lightweight no-op tracer for disabled mode
class _NoOpSpan:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _NoOpTracer:
    def start_as_current_span(self, name: str, *args, **kwargs):
        return _NoOpSpan()

    def _identity_decorator(self, func=None, **kwargs):
        if func is None:
            def _decorator(f):
                return f
            return _decorator
        return func

    # Support usages like @tracer.chain, @tracer.chain(), @tracer.tool, @tracer.agent
    def chain(self, func=None, **kwargs):
        return self._identity_decorator(func, **kwargs)

    def tool(self, func=None, **kwargs):
        return self._identity_decorator(func, **kwargs)

    def agent(self, func=None, **kwargs):
        return self._identity_decorator(func, **kwargs)


# If disabled explicitly or no API key, provide a no-op tracer
if DISABLE_TRACING or not PHOENIX_API_KEY:
    reason = "explicitly disabled via DISABLE_TRACING" if DISABLE_TRACING else "PHOENIX_API_KEY not set"
    print(f"⚠️ Tracing disabled ({reason}); using no-op tracer.")
    tracer = _NoOpTracer()
else:
    # Configure headers for Phoenix
    headers = {"authorization": f"Bearer {PHOENIX_API_KEY}"}
    print("headers: ",headers)
    
    # Register Phoenix tracer with proper configuration
    tracer_provider = register( 
        project_name=f"Agent_{ENV}",
        endpoint=f"{PHOENIX_HOSTNAME}/v1/traces",
        headers=headers,  
        auto_instrument=True,
        verbose=True 
    )
    
    batch_processor = BatchSpanProcessor(
        OTLPSpanExporter(
            endpoint=f"{PHOENIX_HOSTNAME}/v1/traces",
            headers=headers
        ),
        max_queue_size=1024,  #
        max_export_batch_size=128,  
        export_timeout_millis=30000, 
        schedule_delay_millis=2000   
    )


    tracer_provider.add_span_processor(batch_processor)

    # Print Phoenix project info from TracerProvider
    active_processor = tracer_provider._active_span_processor
    processor_type = type(active_processor).__name__

    print("🔭 OpenTelemetry Tracing Details 🔭")
    print(f"|  Phoenix Project: Agent_{ENV}")
    print(f"|  Collector Endpoint: {PHOENIX_HOSTNAME}/v1/traces")
    print(f"|  Span Processor: {processor_type}")
    print(f"|  Transport: HTTP + protobuf")

    # Get processor details
    if hasattr(active_processor, '_span_processors'):
        # It's a MultiSpanProcessor, check what's inside
        processors = [type(p).__name__ for p in active_processor._span_processors]
        print(f"|  Processors: {', '.join(processors)}")
        # Check if BatchSpanProcessor is among them
        if 'BatchSpanProcessor' in processors:
            print(f"|  ✅ BatchSpanProcessor configured for production")
        else:
            print(f"|  ⚠️  No BatchSpanProcessor found - consider adding one for production")
    elif processor_type == "BatchSpanProcessor":
        print(f"|  ✅ Using BatchSpanProcessor for production")
    else:
        print(f"|  ⚠️  Using {processor_type} - consider BatchSpanProcessor for production")

    # Instrument LlamaIndex and OpenAI
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

    # Create the tracer instance
    tracer = tracer_provider.get_tracer(__name__)

    def export_check():
        with tracer.start_as_current_span("connectivity_check"):
            pass
        try:
            tracer_provider.force_flush(timeout_millis=5000)
            return True
        except Exception as e:
            print(f"Export failed: (e)")
            return False

    if not export_check():
        print("⚠️  Warning: Tracer export check failed. Check Tracer configuration.")

    print("✅ Phoenix tracer configured and ready")

