# Load Testing

DocuMind AI includes a lightweight load test script for the chat API.

## What It Measures

- End-to-end request latency
- Success and failure rates
- Request throughput under concurrency
- Approximate p50, p95, and p99 latency

## How To Run

```powershell
python scripts/load_test.py --base-url http://localhost:8000 --requests 50 --concurrency 5
```

## Suggested Baseline Targets

- Backend health under load stays available
- p95 latency remains stable for demo-scale document sets
- Cache hit rate improves on repeated prompts
- Failure rate stays at zero for healthy environments

## When To Use It

- Before releasing a new retrieval configuration
- Before demoing to an investor or customer
- After changing cache, vector store, or LLM settings

## Follow-Up

Use the admin cockpit and telemetry traces to explain any regressions discovered during the test.