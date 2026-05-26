# Demo Runbook

This runbook is optimized for stakeholder demos, portfolio reviews, and investor walkthroughs.

## Demo Goal

Show that DocuMind AI is more than a chat app: it is a production-aware document intelligence system with telemetry, retrieval explainability, and operational controls.

## Suggested Flow

1. Start with the hero screen in the frontend.
2. Upload a document and show the ingestion response.
3. Ask a grounded question and highlight citations.
4. Open the Admin Cockpit and show usage, cache posture, and feature flags.
5. Run a benchmark and show leaderboard/history.
6. Point to the deployment and load-testing docs as proof of production readiness.

## Speaking Points

- Persistent vector storage keeps uploaded knowledge available across sessions
- Retrieval traces explain why an answer was produced
- Metrics and telemetry are exposed without needing to inspect logs manually
- Feature flags and caching let the team control rollout risk
- Billing and quota scaffolding are already tied to real token accounting

## Demo Validation Checklist

- Backend health endpoint is green
- Redis cache is available or explicitly shown as a fallback
- Admin API key is configured
- At least one document is uploaded
- Retrieval and benchmark data are populated

## Scripted Follow-Up

After the live demo, run the enterprise demo script and load test script to show reproducibility.