from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, TYPE_CHECKING
from urllib.parse import quote_plus

import requests

from backend.agentic.models import ToolObservation
if TYPE_CHECKING:
    from backend.services.rag_service import RagService


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_title = False
        self.title = ""
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self.in_title = True

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False

    def handle_data(self, data):
        text = data.strip()
        if not text:
            return
        if self.in_title:
            self.title += text
        elif len(self.parts) < 40:
            self.parts.append(text)


@dataclass(slots=True)
class ToolResult:
    observation: ToolObservation
    citations: list[dict[str, Any]]


class RetrievalTool:
    def __init__(self, rag_service: RagService):
        self.rag_service = rag_service

    def run(self, query: str, top_k: int = 5) -> ToolResult:
        result = self.rag_service.retrieve(query)
        if result is None:
            observation = ToolObservation(tool_name="document_retrieval", query=query, content="No document evidence found.")
            return ToolResult(observation=observation, citations=[])

        context, docs, explanation = result
        citations = [
            {
                "source": doc.metadata.get("source_file"),
                "page": doc.metadata.get("page"),
                "preview": doc.page_content[:300],
            }
            for doc in docs[:top_k]
        ]
        observation = ToolObservation(
            tool_name="document_retrieval",
            query=query,
            content=context,
            metadata={"retrieval_explanation": explanation.model_dump(mode="json") if hasattr(explanation, "model_dump") else explanation},
        )
        return ToolResult(observation=observation, citations=citations)


class WebSearchTool:
    def search(self, query: str, limit: int = 3) -> ToolResult:
        url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
        try:
            response = requests.get(url, timeout=8, headers={"User-Agent": "DocuMind-AI/1.0"})
            response.raise_for_status()
            links = re.findall(r'href="(https?://[^"]+)"', response.text)
            unique_links = []
            for link in links:
                if link not in unique_links and "duckduckgo.com" not in link:
                    unique_links.append(link)
                if len(unique_links) >= limit:
                    break
            citations = [{"url": link} for link in unique_links]
            observation = ToolObservation(tool_name="web_search", query=query, content="\n".join(unique_links), metadata={"result_count": len(unique_links)})
            return ToolResult(observation=observation, citations=citations)
        except Exception as exc:
            observation = ToolObservation(tool_name="web_search", query=query, content=f"Web search unavailable: {exc}")
            return ToolResult(observation=observation, citations=[])


class WebFetchTool:
    def fetch(self, url: str) -> ToolResult:
        try:
            response = requests.get(url, timeout=10, headers={"User-Agent": "DocuMind-AI/1.0"})
            response.raise_for_status()
            parser = _TextExtractor()
            parser.feed(response.text)
            content = " ".join(parser.parts[:80])
            observation = ToolObservation(tool_name="web_fetch", query=url, content=content, metadata={"title": parser.title})
            return ToolResult(observation=observation, citations=[{"url": url, "title": parser.title}])
        except Exception as exc:
            observation = ToolObservation(tool_name="web_fetch", query=url, content=f"Web fetch unavailable: {exc}")
            return ToolResult(observation=observation, citations=[])
