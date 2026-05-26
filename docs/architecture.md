# Architecture

DocuMind AI is organized around a clear separation between the user experience, the retrieval engine, and the production control plane.

## Core Flow

```mermaid
flowchart TB
    Ingest[Document Ingestion] --> Split[Text / OCR / Multimodal Splitter]
    Split --> Embed[Embedding Generation]
    Embed --> Store[Vector Store]
    Store --> Retrieve[Retrieval Service]
    Retrieve --> RAG[RAG / Agentic RAG]
    RAG --> Answer[Answer + Citations]
```

## Control Plane

```mermaid
flowchart LR
    Metrics[Metrics Service] --> Cockpit[Admin Cockpit]
    Telemetry[Telemetry Service] --> Cockpit
    Cache[Redis / Cache Service] --> Cockpit
    DB[(Database)] --> Cockpit
    Flags[Feature Flags] --> Cockpit
    Bench[Benchmark Results] --> Cockpit
    Cockpit --> Usage[Usage + Quota Scaffold]
    Cockpit --> Experiments[A/B Retrieval Experiments]
```

## Production Signals

- Request, token, and latency counters are recorded on the backend
- Retrieval traces and benchmark history are persisted for auditability
- Cache-aware retrieval and streaming paths reduce repeated work
- Feature flags allow agentic RAG, hybrid retrieval, metrics, and Redis cache rollout control

## Investor-Ready Narrative

1. Document upload and ingestion are persistent.
2. Retrieval is explainable with source citations and traceability.
3. Operational metrics are visible in the admin cockpit.
4. The platform is ready for quota, billing, and experiment rollouts.