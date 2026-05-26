from __future__ import annotations

import time
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from starlette.types import ASGIApp, Receive, Scope, Send

from backend.core.metrics import REQUEST_COUNT, REQUEST_LATENCY


class PrometheusMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "-")
        path = scope.get("path", "-")
        start = time.time()

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status = message["status"]
                duration = time.time() - start
                REQUEST_COUNT.labels(method=method, endpoint=path, http_status=str(status)).inc()
                REQUEST_LATENCY.labels(method=method, endpoint=path).observe(duration)
            await send(message)

        await self.app(scope, receive, send_wrapper)


def metrics_endpoint(request):
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
