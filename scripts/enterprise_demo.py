from __future__ import annotations

import argparse
import os
from typing import Any

import requests


def _request(session: requests.Session, method: str, url: str, *, headers: dict[str, str] | None = None, json: dict[str, Any] | None = None) -> dict[str, Any]:
    response = session.request(method=method, url=url, headers=headers, json=json, timeout=60)
    response.raise_for_status()
    return response.json()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a DocuMind AI enterprise demo smoke test.")
    parser.add_argument("--base-url", default=os.getenv("DOCUMIND_API_BASE_URL", "http://localhost:8000"), help="Backend base URL")
    parser.add_argument("--admin-api-key", default=os.getenv("DOCUMIND_ADMIN_API_KEY", ""), help="Admin API key")
    parser.add_argument("--question", default="What are the main obligations in the uploaded documents?", help="Demo question to ask")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    headers = {"X-API-Key": args.admin_api_key} if args.admin_api_key else {}

    session = requests.Session()
    health = _request(session, "GET", f"{base_url}/health")
    metrics = _request(session, "GET", f"{base_url}/admin/metrics", headers=headers)
    debug_state = _request(session, "GET", f"{base_url}/admin/debug/state", headers=headers)

    print("DocuMind AI demo smoke test")
    print(f"Health: {health.get('status', 'ok')}")
    print(f"Vector store ready: {health.get('vector_store_ready')}")
    print(f"Metrics counters: {list((metrics or {}).get('counters', {}).keys())}")
    print(f"Redis available: {debug_state.get('redis_ready')}")
    print(f"Feature flags: {debug_state.get('feature_flags', {})}")

    if debug_state.get("vector_store_ready"):
        answer = _request(session, "POST", f"{base_url}/chat/ask", json={"question": args.question})
        print(f"Answer preview: {str(answer.get('answer', ''))[:240]}")
    else:
        print("Answer preview: skipped because the vector store is not ready yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())