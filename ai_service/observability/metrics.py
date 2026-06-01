from prometheus_client import Counter, Histogram

# Total AI requests handled
ai_requests_total = Counter(
    "ai_requests_total",
    "Total number of AI requests processed",
    ["status"]  # "success" or "error"
)

# Total tool calls executed
ai_tool_calls_total = Counter(
    "ai_tool_calls_total",
    "Total number of tool executions",
    ["tool_name", "success"]  # success: "true" or "false"
)

# Total provider failovers
ai_provider_failover_total = Counter(
    "ai_provider_failover_total",
    "Total number of provider failover events",
    ["primary_provider", "fallback_provider"]
)

# Total tokens consumed
ai_tokens_total = Counter(
    "ai_tokens_total",
    "Total number of LLM tokens consumed",
    ["model", "token_type"]  # token_type: "prompt" or "completion"
)

# AI request response latency
ai_latency_seconds = Histogram(
    "ai_latency_seconds",
    "Latency of AI chat responses in seconds",
    ["model"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
)

# Tool execution latency
ai_tool_latency_seconds = Histogram(
    "ai_tool_latency_seconds",
    "Latency of tool executions in seconds",
    ["tool_name"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
)
