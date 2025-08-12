"""

"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional
import uuid
import time
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes


# Initialize OpenTelemetry
resource = Resource(attributes={
    ResourceAttributes.SERVICE_NAME: "agentic-ai-service",
    ResourceAttributes.SERVICE_VERSION: "1.0.0",
    "environment": "production"
})

# Create a meter provider
metric_reader = PeriodicExportingMetricReader(ConsoleMetricExporter())
meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
metrics.set_meter_provider(meter_provider)
meter = metrics.get_meter("ai.agent.metrics")

class AIMetricType(Enum):
    """Types of AI-related metrics."""
    LLM_CALL = "llm.call"
    NODE_EXECUTION = "node.execution"
    TOOL_CALL = "tool.call"
    CHAIN_EXECUTION = "chain.execution"

class AIModelProvider(Enum):
    """Supported model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    VERTEX_AI = "vertex_ai"
    BEDROCK = "bedrock"
    HUGGINGFACE = "huggingface"

class AIModelType(Enum):
    """Types of AI models."""
    COMPLETION = "completion"
    CHAT = "chat"
    EMBEDDING = "embedding"
    IMAGE = "image"

@dataclass
class AIMetricAttributes:
    """Common attributes for AI metrics following OpenTelemetry semantic conventions."""
    # Required attributes
    metric_type: AIMetricType
    model_provider: AIModelProvider
    model_name: str
    model_type: AIModelType
    
    # Optional attributes
    model_version: Optional[str] = None
    environment: str = "production"
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    chain_id: Optional[str] = None
    node_id: Optional[str] = None
    tool_name: Optional[str] = None
    error: bool = False
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    
    # Additional dimensions
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_attributes(self) -> Dict[str, str]:
        """Convert to OpenTelemetry attributes."""
        attrs = {
            "ai.metric.type": self.metric_type.value,
            "ai.model.provider": self.model_provider.value,
            "ai.model.name": self.model_name,
            "ai.model.type": self.model_type.value,
            "environment": self.environment,
            "error": str(self.error).lower(),
        }
        
        # Add optional fields if they exist
        if self.model_version:
            attrs["ai.model.version"] = self.model_version
        if self.user_id:
            attrs["enduser.id"] = self.user_id
        if self.session_id:
            attrs["session.id"] = self.session_id
        if self.request_id:
            attrs["request.id"] = self.request_id
        if self.chain_id:
            attrs["ai.chain.id"] = self.chain_id
        if self.node_id:
            attrs["ai.node.id"] = self.node_id
        if self.tool_name:
            attrs["ai.tool.name"] = self.tool_name
        if self.error and self.error_type:
            attrs["error.type"] = self.error_type
        if self.error and self.error_message:
            attrs["error.message"] = self.error_message
            
        # Add custom tags
        attrs.update(self.tags)
        return attrs

class AIMetrics:
    """Metrics collector for AI operations following OpenTelemetry semantic conventions."""
    
    def __init__(self):
        # LLM Metrics
        self.llm_requests = meter.create_counter(
            name="ai.llm.requests",
            unit="1",
            description="Count of LLM API requests"
        )
        
        self.llm_prompt_tokens = meter.create_histogram(
            name="ai.llm.tokens.prompt",
            unit="tokens",
            description="Number of tokens in prompts"
        )
        
        self.llm_completion_tokens = meter.create_histogram(
            name="ai.llm.tokens.completion",
            unit="tokens",
            description="Number of tokens in completions"
        )
        
        self.llm_errors = meter.create_counter(
            name="ai.llm.errors",
            unit="1",
            description="Count of LLM API errors"
        )
        
        # Node Execution Metrics
        self.node_executions = meter.create_counter(
            name="ai.node.executions",
            unit="1",
            description="Count of node executions"
        )
        
        self.node_duration = meter.create_histogram(
            name="ai.node.duration",
            unit="ms",
            description="Duration of node executions in milliseconds"
        )
        
        # Tool Call Metrics
        self.tool_calls = meter.create_counter(
            name="ai.tool.calls",
            unit="1",
            description="Count of tool calls"
        )
        
        self.tool_duration = meter.create_histogram(
            name="ai.tool.duration",
            unit="ms",
            description="Duration of tool executions in milliseconds"
        )
        
        # Chain Execution Metrics
        self.chain_executions = meter.create_counter(
            name="ai.chain.executions",
            unit="1",
            description="Count of chain executions"
        )
        
        self.chain_duration = meter.create_histogram(
            name="ai.chain.duration",
            unit="ms",
            description="Duration of chain executions in milliseconds"
        )
        
        # Common Metrics
        self.request_latency = meter.create_histogram(
            name="ai.request.latency",
            unit="ms",
            description="Latency of AI operations"
        )
        
        self.token_usage = meter.create_counter(
            name="ai.tokens.total",
            unit="tokens",
            description="Total token usage"
        )
    
    def record_llm_call(
        self,
        attributes: AIMetricAttributes,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        duration_ms: float = 0.0
    ) -> None:
        """Record metrics for an LLM API call."""
        attrs = attributes.to_attributes()
        
        # Record request
        self.llm_requests.add(1, attrs)
        
        # Record token usage
        if prompt_tokens > 0:
            self.llm_prompt_tokens.record(prompt_tokens, attrs)
        if completion_tokens > 0:
            self.llm_completion_tokens.record(completion_tokens, attrs)
        if total_tokens > 0:
            self.token_usage.add(total_tokens, attrs)
        
        # Record latency
        if duration_ms > 0:
            self.request_latency.record(duration_ms, attrs)
        
        # Record error if any
        if attributes.error:
            self.llm_errors.add(1, attrs)
    
    def record_node_execution(
        self,
        attributes: AIMetricAttributes,
        duration_ms: float,
        success: bool = True
    ) -> None:
        """Record metrics for a node execution."""
        attrs = attributes.to_attributes()
        attrs["success"] = str(success).lower()
        
        self.node_executions.add(1, attrs)
        self.node_duration.record(duration_ms, attrs)
        
        if not success:
            self.record_error(attributes)
    
    def record_tool_call(
        self,
        attributes: AIMetricAttributes,
        duration_ms: float,
        success: bool = True
    ) -> None:
        """Record metrics for a tool call."""
        attrs = attributes.to_attributes()
        attrs["success"] = str(success).lower()
        
        self.tool_calls.add(1, attrs)
        self.tool_duration.record(duration_ms, attrs)
        
        if not success:
            self.record_error(attributes)
    
    def record_chain_execution(
        self,
        attributes: AIMetricAttributes,
        duration_ms: float,
        success: bool = True
    ) -> None:
        """Record metrics for a chain execution."""
        attrs = attributes.to_attributes()
        attrs["success"] = str(success).lower()
        
        self.chain_executions.add(1, attrs)
        self.chain_duration.record(duration_ms, attrs)
        
        if not success:
            self.record_error(attributes)
    
    def record_error(self, attributes: AIMetricAttributes) -> None:
        """Record an error metric."""
        attrs = attributes.to_attributes()
        self.llm_errors.add(1, attrs)

# Example usage
if __name__ == "__main__":
    # Initialize metrics
    metrics = AIMetrics()
    
    # Example LLM call
    llm_attrs = AIMetricAttributes(
        metric_type=AIMetricType.LLM_CALL,
        model_provider=AIModelProvider.OPENAI,
        model_name="gpt-4",
        model_type=AIModelType.CHAT,
        model_version="0613",
        request_id=str(uuid.uuid4()),
        user_id="user123",
        tags={"deployment": "production-us-central1"}
    )
    
    start_time = time.time()
    try:
        # Simulate LLM call
        time.sleep(0.1)
        
        # Record successful LLM call
        metrics.record_llm_call(
            attributes=llm_attrs,
            prompt_tokens=150,
            completion_tokens=50,
            total_tokens=200,
            duration_ms=(time.time() - start_time) * 1000
        )
    except Exception as e:
        # Record failed LLM call
        error_attrs = AIMetricAttributes(
            **llm_attrs.__dict__,
            error=True,
            error_type=type(e).__name__,
            error_message=str(e)
        )
        metrics.record_llm_call(
            attributes=error_attrs,
            duration_ms=(time.time() - start_time) * 1000
        )
    
    # Example chain execution
    chain_attrs = AIMetricAttributes(
        metric_type=AIMetricType.CHAIN_EXECUTION,
        model_provider=AIModelProvider.OPENAI,
        model_name="gpt-4",
        model_type=AIModelType.CHAT,
        chain_id="customer_support_chain",
        request_id=str(uuid.uuid4())
    )
    
    start_time = time.time()
    try:
        # Simulate chain execution
        time.sleep(0.2)
        
        # Record successful chain execution
        metrics.record_chain_execution(
            attributes=chain_attrs,
            duration_ms=(time.time() - start_time) * 1000
        )
    except Exception as e:
        # Record failed chain execution
        error_attrs = AIMetricAttributes(
            **chain_attrs.__dict__,
            error=True,
            error_type=type(e).__name__,
            error_message=str(e)
        )
        metrics.record_chain_execution(
            attributes=error_attrs,
            duration_ms=(time.time() - start_time) * 1000
        )
        
from opentelemetry.sdk.metrics import MeterProvider, View
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics.aggregation import ExplicitBucketHistogramAggregation

views = [
    View(
        instrument_name="gen_ai.client.token.usage",
        aggregation=ExplicitBucketHistogramAggregation([1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864]),
    ),
    View(
        instrument_name="gen_ai.client.operation.duration",
        aggregation=ExplicitBucketHistogramAggregation([0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92]),
    ),
]

metric_exporter = OTLPMetricExporter(endpoint="http://localhost:4317")
metric_reader = PeriodicExportingMetricReader(metric_exporter)
provider = MeterProvider(
    metric_readers=[metric_reader],
    views=views
)

from opentelemetry.sdk.metrics import set_meter_provider
set_meter_provider(provider)