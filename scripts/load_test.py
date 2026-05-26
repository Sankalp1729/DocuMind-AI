from __future__ import annotations

import argparse
import os
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


PROMPTS = [
    "Summarize the key obligations in the document.",
    "What deadlines are mentioned in the file?",
    "List the core risks described in the policy.",
    "Which sections reference compliance requirements?",
    "Explain the main decision criteria from the text.",
]


def _ask(base_url: str, question: str, session_id: str) -> float:
    session = requests.Session()
    start = time.perf_counter()
    response = session.post(
        f"{base_url}/chat/ask",
        json={"question": question},
        headers={"X-Session-ID": session_id},
        timeout=120,
    )
    response.raise_for_status()
    return (time.perf_counter() - start) * 1000


def main() -> int:
    parser = argparse.ArgumentParser(description="Load test the DocuMind AI chat endpoint.")
    parser.add_argument("--base-url", default=os.getenv("DOCUMIND_API_BASE_URL", "http://localhost:8000"), help="Backend base URL")
    parser.add_argument("--requests", type=int, default=int(os.getenv("DOCUMIND_LOAD_TEST_REQUESTS", "20")), help="Total requests to run")
    parser.add_argument("--concurrency", type=int, default=int(os.getenv("DOCUMIND_LOAD_TEST_CONCURRENCY", "4")), help="Concurrent workers")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    latencies: list[float] = []
    failures = 0

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = []
        for index in range(args.requests):
            question = random.choice(PROMPTS)
            session_id = f"loadtest-{index % max(args.concurrency, 1)}"
            futures.append(executor.submit(_ask, base_url, question, session_id))

        for future in as_completed(futures):
            try:
                latencies.append(future.result())
            except Exception:
                failures += 1

    elapsed = time.perf_counter() - start
    total = len(latencies) + failures
    throughput = total / elapsed if elapsed else 0.0

    if latencies:
        p50 = statistics.median(latencies)
        p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)
        p99 = max(latencies)
    else:
        p50 = p95 = p99 = 0.0

    print("DocuMind AI load test")
    print(f"Requests: {total}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Failures: {failures}")
    print(f"Throughput req/s: {throughput:.2f}")
    print(f"p50 latency ms: {p50:.1f}")
    print(f"p95 latency ms: {p95:.1f}")
    print(f"p99 latency ms: {p99:.1f}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())