from prometheus_client import Counter, Histogram

# HTTP request metrics
REQUEST_COUNT = Counter(
    "documind_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "http_status"],
)

REQUEST_LATENCY = Histogram(
    "documind_http_request_latency_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
)

# Retrieval and RAG metrics
RETRIEVAL_REQUESTS = Counter("documind_retrieval_requests_total", "Total retrieval requests", ["origin"])
RETRIEVAL_LATENCY = Histogram("documind_retrieval_latency_seconds", "Retrieval latency seconds", [])

# Token / model usage metrics
TOKEN_USAGE = Counter("documind_tokens_total", "Total tokens consumed", ["model", "type"])  # type = prompt|completion
