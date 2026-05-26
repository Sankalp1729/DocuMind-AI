from __future__ import annotations

import io
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import requests
from dotenv import load_dotenv


load_dotenv()
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ApiConfig:
    base_url: str = os.getenv("DOCUMIND_API_BASE_URL", "http://localhost:8000")
    timeout_seconds: int = int(os.getenv("DOCUMIND_API_TIMEOUT_SECONDS", "120"))
    max_retries: int = int(os.getenv("DOCUMIND_API_MAX_RETRIES", "2"))
    backoff_factor: float = float(os.getenv("DOCUMIND_API_BACKOFF_FACTOR", "0.5"))


class ApiClientError(RuntimeError):
    pass


class ApiClient:
    def __init__(self, base_url: str | None = None, config: ApiConfig | None = None, session_id: str | None = None, admin_api_key: str | None = None):
        self.config = config or ApiConfig(base_url=base_url or ApiConfig.base_url)
        if base_url:
            self.config.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session_id = session_id
        self.admin_api_key = admin_api_key

    @property
    def base_url(self) -> str:
        return self.config.base_url.rstrip("/")

    def _endpoint(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _parse_error(self, response: requests.Response) -> str:
        try:
            payload = response.json()
            return payload.get("detail") or payload.get("message") or response.text
        except ValueError:
            return response.text or f"HTTP {response.status_code}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        files=None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        attempts = self.config.max_retries + 1
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                start = time.perf_counter()
                response = self.session.request(
                    method=method,
                    url=self._endpoint(path),
                    json=json,
                    files=files,
                    timeout=self.config.timeout_seconds,
                    headers={
                        **({"X-Session-ID": self.session_id} if self.session_id else {}),
                        **(headers or {}),
                    } or None,
                )
                elapsed_ms = int((time.perf_counter() - start) * 1000)

                if response.status_code >= 400:
                    raise ApiClientError(f"{self._parse_error(response)} (HTTP {response.status_code})")

                try:
                    payload = response.json()
                except ValueError as exc:
                    raise ApiClientError("Backend returned malformed JSON") from exc

                payload["_latency_ms"] = elapsed_ms
                return payload
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                last_error = exc
                logger.warning("API request attempt %s/%s failed: %s", attempt, attempts, exc)
                if attempt == attempts:
                    break
                time.sleep(self.config.backoff_factor * attempt)
            except ApiClientError:
                raise
            except requests.RequestException as exc:
                raise ApiClientError(f"Unexpected API request failure: {exc}") from exc

        raise ApiClientError(f"Backend unavailable after {attempts} attempt(s): {last_error}")

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def upload_document(self, uploaded_file) -> dict[str, Any]:
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name
        mime_type = getattr(uploaded_file, "type", None) or "application/pdf"

        file_buffer = io.BytesIO(file_bytes)
        files = {"file": (file_name, file_buffer, mime_type)}
        return self._request("POST", "/documents/upload", files=files)

    def ask_question(self, question: str) -> dict[str, Any]:
        return self._request("POST", "/chat/ask", json={"question": question})

    def _admin_headers(self) -> dict[str, str]:
        if not self.admin_api_key:
            return {}
        return {"X-API-Key": self.admin_api_key}

    def admin_metrics(self) -> dict[str, Any]:
        return self._request("GET", "/admin/metrics", headers=self._admin_headers())

    def admin_debug_state(self) -> dict[str, Any]:
        return self._request("GET", "/admin/debug/state", headers=self._admin_headers())

    def admin_retrieval_debug(self, query: str) -> dict[str, Any]:
        return self._request("GET", f"/admin/retrieval-debug?query={requests.utils.quote(query)}", headers=self._admin_headers())

    def admin_retrieval_trace(self, query: str) -> dict[str, Any]:
        return self._request("GET", f"/admin/retrieval-trace?query={requests.utils.quote(query)}", headers=self._admin_headers())

    def admin_benchmarks(self) -> dict[str, Any]:
        return self._request("GET", "/admin/benchmarks", headers=self._admin_headers())

    def admin_evaluate(self, run_id: str | None = None) -> dict[str, Any]:
        path = "/admin/evaluate"
        if run_id:
            path = f"{path}?run_id={requests.utils.quote(run_id)}"
        return self._request("POST", path, headers=self._admin_headers())

    def admin_evaluation_datasets(self) -> dict[str, Any]:
        return self._request("GET", "/admin/evaluation/datasets", headers=self._admin_headers())

    def admin_evaluation_history(self) -> dict[str, Any]:
        return self._request("GET", "/admin/evaluation/history", headers=self._admin_headers())

    def admin_evaluation_leaderboard(self) -> dict[str, Any]:
        return self._request("GET", "/admin/evaluation/leaderboard", headers=self._admin_headers())

    def admin_run_benchmark(self, dataset_name: str, run_id: str | None = None, top_k: int = 10) -> dict[str, Any]:
        path = f"/admin/evaluate?dataset_name={requests.utils.quote(dataset_name)}&top_k={top_k}"
        if run_id:
            path = f"{path}&run_id={requests.utils.quote(run_id)}"
        return self._request("POST", path, headers=self._admin_headers())
