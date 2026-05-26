from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import streamlit as st

from frontend.components.retrieval_debug import render_retrieval_debug, render_groundedness_indicator
from frontend.services.api_client import ApiClient, ApiClientError


def _stat_card(label: str, value: str, caption: str | None = None) -> str:
    caption_html = f'<div class="documind-card-caption">{caption}</div>' if caption else ""
    return f"""
        <div class="documind-stat-card">
            <div class="documind-stat-label">{label}</div>
            <div class="documind-stat-value">{value}</div>
            {caption_html}
        </div>
    """


def _render_stat_row(items: list[tuple[str, str, str | None]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value, caption) in zip(columns, items):
        with column:
            st.markdown(_stat_card(label, value, caption), unsafe_allow_html=True)


def _load_admin_overview(api_client: ApiClient) -> dict[str, Any]:
    return {
        "metrics": api_client.admin_metrics(),
        "debug_state": api_client.admin_debug_state(),
        "loaded_at": datetime.now(timezone.utc).isoformat(),
    }


def _render_benchmark_dashboard(payload: dict) -> None:
    dashboard = payload.get("dashboard") or {}
    benchmarks = payload.get("benchmarks") or []
    leaderboard = dashboard.get("leaderboard") or []
    history = dashboard.get("history") or []

    st.markdown("### Benchmark Dashboard")
    if not dashboard:
        st.info("No benchmark runs have been persisted yet.")
        return

    cols = st.columns(4)
    cols[0].metric("Runs", dashboard.get("runs", 0))

    average_metrics = dashboard.get("average_metrics", {})
    cols[1].metric("Precision@10", f"{average_metrics.get('precision_at_10', 0.0):.3f}")
    cols[2].metric("Recall@10", f"{average_metrics.get('recall_at_10', 0.0):.3f}")
    cols[3].metric("mRR", f"{average_metrics.get('mrr', 0.0):.3f}")

    latest_run = dashboard.get("latest_run") or {}
    if latest_run:
        st.markdown("#### Latest Run")
        st.json(latest_run)

    trend = dashboard.get("trend") or []
    if trend:
        st.markdown("#### Trend")
        st.dataframe(trend, use_container_width=True)

    if leaderboard:
        st.markdown("#### Leaderboard")
        st.dataframe(leaderboard, use_container_width=True)

    if history:
        with st.expander("Evaluation history", expanded=False):
            st.dataframe(history, use_container_width=True)

    if benchmarks:
        with st.expander("Raw benchmark records", expanded=False):
            st.json(benchmarks)


def _render_feature_flags(feature_flags: dict[str, Any]) -> None:
    st.markdown("### Feature Flags")
    if not feature_flags:
        st.info("No feature-flag data available yet.")
        return

    flag_rows = []
    for key, value in feature_flags.items():
        flag_rows.append({"flag": key, "enabled": "On" if value else "Off"})

    st.dataframe(flag_rows, use_container_width=True, hide_index=True)


def _render_usage_scaffold(usage: dict[str, Any]) -> None:
    st.markdown("### Usage Monitoring and Billing Scaffold")
    if not usage:
        st.info("Usage data will appear after conversations, token accounting, and analytics persistence are active.")
        return

    total_tokens = int(usage.get("total_tokens", 0) or 0)
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    estimated_cost = total_tokens / 1000 * 0.002
    quota_limit = 500_000
    quota_used = min(total_tokens / quota_limit, 1.0) if quota_limit else 0.0

    _render_stat_row(
        [
            ("Total tokens", f"{total_tokens:,}", "Cross-session usage aggregate"),
            ("Prompt tokens", f"{prompt_tokens:,}", "Input side accounting"),
            ("Completion tokens", f"{completion_tokens:,}", "Output side accounting"),
            ("Illustrative spend", f"${estimated_cost:,.2f}", "Scaffolding only; connect billing provider later"),
        ]
    )

    st.progress(quota_used)
    st.caption(f"Quota scaffold: {total_tokens:,} / {quota_limit:,} tokens used")

    quota_table = [
        {"plan": "Starter", "monthly_quota_tokens": 100_000, "notes": "Pilot teams and demos"},
        {"plan": "Growth", "monthly_quota_tokens": 500_000, "notes": "Internal enterprise adoption"},
        {"plan": "Enterprise", "monthly_quota_tokens": 2_500_000, "notes": "Custom contract and SSO"},
    ]
    st.dataframe(quota_table, use_container_width=True, hide_index=True)


def _render_experiment_scaffold(debug_state: dict[str, Any]) -> None:
    st.markdown("### A/B Retrieval Experiments")
    retrieval_config = debug_state.get("retrieval_configuration", {})
    feature_flags = debug_state.get("feature_flags", {})

    experiment_table = [
        {
            "variant": "Control",
            "retrieval_stack": f"Current stack, top_k={retrieval_config.get('top_k', 0)}",
            "primary_metric": "Precision@10",
            "guardrail": "Latency and groundedness",
        },
        {
            "variant": "Treatment A",
            "retrieval_stack": "Hybrid retrieval + rerank",
            "primary_metric": "Recall@10",
            "guardrail": "Response latency < 1.5x control",
        },
        {
            "variant": "Treatment B",
            "retrieval_stack": "Cached retrieval with broader top-k",
            "primary_metric": "Token cost per answer",
            "guardrail": "Groundedness remains green",
        },
    ]
    st.dataframe(experiment_table, use_container_width=True, hide_index=True)

    if feature_flags.get("hybrid_retrieval"):
        st.success("Hybrid retrieval is enabled, so the experimental treatment path can be exercised directly.")
    else:
        st.info("Hybrid retrieval is currently disabled; use the treatment rows as rollout scaffolding.")


def _render_performance_snapshot(metrics: dict[str, Any], debug_state: dict[str, Any]) -> None:
    st.markdown("### Performance Optimization")
    counters = metrics.get("counters", {}) if isinstance(metrics, dict) else {}
    latencies = metrics.get("latencies", {}) if isinstance(metrics, dict) else {}
    cache_state = debug_state.get("cache", {})
    retrieval_config = debug_state.get("retrieval_configuration", {})

    _render_stat_row(
        [
            ("Chat requests", f"{int(counters.get('chat_requests_total', 0) or 0):,}", "Backend request counter"),
            ("Metric families", f"{len(counters):,}", "Custom counters captured in memory"),
            ("Latency groups", f"{len(latencies):,}", "Tracked latency aggregates"),
            ("Cache", "Warm" if cache_state.get("available") else "Fallback", "Response and retrieval cache posture"),
        ]
    )

    optimization_table = [
        {"control": "Response caching", "status": "Enabled" if cache_state.get("available") else "Unavailable", "notes": "Conversation and streaming caches are already wired."},
        {"control": "Embedding caching", "status": "Enabled", "notes": "Embeddings are cached at process scope in the vector store layer."},
        {"control": "Retrieval depth", "status": f"top_k={retrieval_config.get('top_k', 0)}", "notes": "Tune this per workload and experiment variant."},
        {"control": "Agentic RAG", "status": "Enabled" if debug_state.get("feature_flags", {}).get("agentic_rag") else "Disabled", "notes": "Use for premium workflows and long-form answers."},
    ]
    st.dataframe(optimization_table, use_container_width=True, hide_index=True)


def render_diagnostics(api_client: ApiClient) -> None:
    st.markdown('<div class="documind-section-title">Admin Analytics Cockpit</div>', unsafe_allow_html=True)
    st.markdown('<div class="documind-subtitle">Monitor usage, inspect retrieval quality, review cache posture, and stage billing and experiment controls.</div>', unsafe_allow_html=True)

    refresh_col, status_col = st.columns([1, 3])
    with refresh_col:
        refresh_dashboard = st.button("Refresh cockpit", use_container_width=True)
    with status_col:
        st.info("The cockpit combines backend metrics, debug state, and persisted evaluation history in one place.")

    if refresh_dashboard or "latest_admin_dashboard" not in st.session_state:
        try:
            st.session_state.latest_admin_dashboard = _load_admin_overview(api_client)
            st.session_state.admin_dashboard_error = None
        except ApiClientError as exc:
            st.session_state.admin_dashboard_error = str(exc)
            st.session_state.latest_admin_dashboard = None

    if st.session_state.get("admin_dashboard_error"):
        st.error(st.session_state.admin_dashboard_error)
        return

    dashboard = st.session_state.get("latest_admin_dashboard") or {}
    metrics = dashboard.get("metrics") or {}
    debug_state = dashboard.get("debug_state") or {}
    counters = metrics.get("counters", {}) if isinstance(metrics, dict) else {}
    usage = debug_state.get("usage", {})
    feature_flags = debug_state.get("feature_flags", {})
    cache_state = debug_state.get("cache", {})

    _render_stat_row(
        [
            ("Chat requests", f"{int(counters.get('chat_requests_total', 0) or 0):,}", "User-facing requests tracked by metrics"),
            ("Tokens", f"{int(usage.get('total_tokens', 0) or 0):,}", "All recorded prompt and completion tokens"),
            ("Retrieval traces", f"{int(usage.get('retrieval_traces', 0) or 0):,}", "Persisted traces for debugging and audit"),
            ("Cache posture", "Ready" if cache_state.get("available") else "Fallback", "Redis-backed when available"),
        ]
    )

    left_col, right_col = st.columns([1.25, 1])
    with left_col:
        st.markdown("### Live Metrics")
        st.json(metrics)

    with right_col:
        st.markdown("### Platform Health")
        health_rows = [
            {"system": "Vector store", "status": "Ready" if debug_state.get("vector_store_ready") else "Empty"},
            {"system": "BM25", "status": "Loaded" if debug_state.get("bm25_loaded") else "Idle"},
            {"system": "Redis", "status": "Online" if debug_state.get("redis_ready") else "Fallback"},
            {"system": "Database", "status": "Connected" if debug_state.get("database_ready") else "Unavailable"},
        ]
        st.dataframe(health_rows, use_container_width=True, hide_index=True)

    _render_performance_snapshot(metrics, debug_state)
    _render_usage_scaffold(usage)
    _render_feature_flags(feature_flags)
    _render_experiment_scaffold(debug_state)

    st.markdown('<div class="documind-section-title">Benchmark Operations</div>', unsafe_allow_html=True)
    try:
        datasets_payload = api_client.admin_evaluation_datasets()
        datasets = datasets_payload.get("datasets", [])
    except ApiClientError as exc:
        datasets = []
        st.warning(f"Unable to load benchmark datasets: {exc}")

    query = st.text_input(
        "Debug query",
        key="diagnostics_query",
        placeholder="What does the system extract from documents?",
        help="Use this to inspect retrieval traces and the grounding path behind an answer.",
    )

    dataset_name = None
    if datasets:
        dataset_name = st.selectbox(
            "Benchmark dataset",
            options=[dataset.get("dataset_name") for dataset in datasets],
            index=0,
            help="Datasets are stored under data/evaluation_datasets/.",
        )
    else:
        dataset_name = st.text_input("Benchmark dataset", value="documind_baseline", help="Fallback when no persisted datasets are present.")

    col1, col2, col3 = st.columns(3)
    run_debug = col1.button("Run Retrieval Debug", use_container_width=True)
    run_trace = col2.button("Load Trace + Benchmarks", use_container_width=True)
    run_benchmark = col3.button("Run Benchmark", use_container_width=True)

    if run_debug:
        if not query.strip():
            st.warning("Enter a query first.")
        else:
            try:
                st.session_state.latest_retrieval_debug = api_client.admin_retrieval_debug(query)
                st.success("Loaded retrieval debug payload.")
            except ApiClientError as exc:
                st.error(str(exc))

    if run_trace:
        if not query.strip():
            st.warning("Enter a query first.")
        else:
            try:
                st.session_state.latest_retrieval_trace = api_client.admin_retrieval_trace(query)
                st.session_state.latest_benchmarks = api_client.admin_benchmarks()
                st.success("Loaded trace and benchmark data.")
            except ApiClientError as exc:
                st.error(str(exc))

    if run_benchmark:
        try:
            benchmark_payload = api_client.admin_run_benchmark(dataset_name or "documind_baseline")
            st.session_state.latest_benchmark_run = benchmark_payload
            st.session_state.latest_benchmarks = api_client.admin_benchmarks()
            st.session_state.latest_evaluation_history = api_client.admin_evaluation_history()
            st.session_state.latest_leaderboard = api_client.admin_evaluation_leaderboard()
            st.success(f"Benchmark run completed for {dataset_name}.")
        except ApiClientError as exc:
            st.error(str(exc))

    retrieval_debug = st.session_state.get("latest_retrieval_debug")
    if retrieval_debug:
        st.markdown("### Latest Retrieval Debug")
        st.json({
            "query": retrieval_debug.get("query"),
            "num_results": retrieval_debug.get("num_results"),
            "trace": retrieval_debug.get("trace"),
        })
        render_retrieval_debug(retrieval_debug.get("retrieval_explanation"))
        trace = retrieval_debug.get("trace") or {}
        render_groundedness_indicator(trace.get("groundedness"))

    retrieval_trace = st.session_state.get("latest_retrieval_trace")
    if retrieval_trace:
        st.markdown("### Telemetry Trace")
        st.json(retrieval_trace)

    benchmark_run = st.session_state.get("latest_benchmark_run")
    if benchmark_run:
        st.markdown("### Latest Benchmark Run")
        st.json(benchmark_run)

    benchmarks = st.session_state.get("latest_benchmarks")
    if benchmarks:
        _render_benchmark_dashboard(benchmarks)

    history_payload = st.session_state.get("latest_evaluation_history")
    if history_payload:
        st.markdown("### Evaluation History")
        st.dataframe(history_payload.get("history", []), use_container_width=True)

    leaderboard_payload = st.session_state.get("latest_leaderboard")
    if leaderboard_payload:
        st.markdown("### Leaderboard")
        st.dataframe(leaderboard_payload.get("leaderboard", []), use_container_width=True)
