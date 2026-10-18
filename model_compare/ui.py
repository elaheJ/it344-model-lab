"""Shared Streamlit interface for the Python and checked-agent apps."""
import inspect
import io
import json
import os
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from . import core
from .agent import context_id, http_controller, local_controller, rehearsal_controller, run_agent
from .snapshots import demo_snapshot, default_snapshot, load_snapshot
from .live_data import SnapshotFeed

DATA = Path(__file__).parent / "data"
LABELS = {k: value[0] for k, value in core.METRICS.items()}


def settings(name, default=""):
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return os.getenv(name, default)


def json_download(label, value, filename, key=None):
    st.download_button(label, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                       file_name=filename, mime="application/json", key=key)


def setup(agent):
    st.set_page_config(page_title="Model Lab · " + ("Agent" if agent else "Compare"), page_icon="◈", layout="wide")
    st.markdown("""<style>
    .stApp {background:#fff;}
    .block-container {padding-top:2.6rem; max-width:1250px;}
    h1,h2,h3 {letter-spacing:-.035em; color:#111;}
    h1 {font-size:2.5rem !important; font-weight:650 !important;}
    [data-testid="stSidebar"] {background:#f7f7f7; border-right:1px solid #ececec;}
    [data-testid="stMetric"] {padding:12px 0; border-bottom:1px solid #e5e5e5;}
    [data-testid="stMetricValue"] {font-size:1.35rem; white-space:normal;}
    [data-testid="stMetricValue"] > div {white-space:normal; overflow-wrap:anywhere;}
    [data-testid="stMetricLabel"] p {white-space:normal;}
    .eyebrow {color:#cc0000;font-size:.72rem;font-weight:700;letter-spacing:.15em;margin-bottom:.8rem;}
    [data-testid="stAppDeployButton"] {display:none;}
    </style>""", unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">ILLINOIS STATE UNIVERSITY / IT344</div>', unsafe_allow_html=True)
    st.title("Model Lab" + (" · Agent" if agent else ""))
    st.caption("Compare models. Find alternatives." + (" Let an agent make a checked choice." if agent else ""))


@st.cache_resource(show_spinner=False)
def shared_feed():
    return SnapshotFeed()


def snapshot_controls():
    with st.sidebar:
        st.subheader("Data")
        with st.expander("Use a different snapshot"):
            upload = st.file_uploader("Load JSON or CSV", type=["json", "csv"], help="Maximum 10 MB.")
            use_demo = st.checkbox("Use fictional demo data", value=False)
            st.download_button("CSV template", (DATA / "snapshot_template.csv").read_bytes(), "snapshot_template.csv", "text/csv")
        live, unavailable = None, False
        try:
            if use_demo:
                snapshot = demo_snapshot()
            elif upload:
                snapshot = load_snapshot(upload.getvalue(), upload.name)
            else:
                api_key = settings("AA_API_KEY") or settings("ARTIFICIAL_ANALYSIS_API_KEY")
                if api_key:
                    with st.spinner("Loading model data…"):
                        live, unavailable = shared_feed().read(api_key.strip())
                snapshot = live or default_snapshot()
        except (ValueError, OSError) as exc:
            st.error(str(exc))
            st.stop()
        if unavailable:
            st.caption("Live refresh is temporarily unavailable. Using the last saved snapshot.")
        label = "Fictional demo" if snapshot["is_demo"] else "Latest AA data" if live else "Loaded snapshot"
        st.caption(f"{label} · {len(snapshot['models'])} models")
        with st.expander("Source details"):
            st.write("Source:", snapshot["source"])
            st.write("Retrieved:", snapshot["retrieved_at"])
            st.write("Index version:", snapshot["intelligence_index_version"])
            st.caption("SHA-256: " + snapshot["snapshot_id"])
            json_download("Download snapshot", snapshot, "snapshot.json")
    return snapshot


def task_controls(snapshot, agent):
    with st.sidebar:
        with st.expander("Models"):
            creators = sorted({m["model_creator"]["name"] for m in snapshot["models"]})
            creator_filter = st.multiselect("Creators (empty = all)", creators)
            search = st.text_input("Search model name or ID")
            models = [m for m in snapshot["models"] if (not creator_filter or m["model_creator"]["name"] in creator_filter)
                      and search.lower() in (m["name"] + " " + m["slug"]).lower()]
            names = {m["slug"]: m["name"] for m in models}
            slugs = st.multiselect("Models to compare", list(names), default=list(names)[:8],
                                   format_func=lambda s: f"{names.get(s, s)} · {s}", key="candidates_" + snapshot["snapshot_id"][:12] + "_" + str(len(models)))
        if agent and len(slugs) > 12:
            st.warning("Select at most 12 candidates for the agent.")
            st.stop()
        with st.expander("Workload"):
            inp = st.number_input("Input tokens", min_value=0, max_value=10**12, value=100_000, step=10_000)
            out = st.number_input("Output tokens (including reasoning)", min_value=0, max_value=10**12, value=20_000, step=1_000)
        with st.expander("Requirements"):
            quality = st.number_input("Minimum intelligence index", min_value=0.0, value=30.0, step=1.0)
            ttft = st.number_input("Maximum first-token delay (s)", min_value=0.0, value=10.0, step=0.1)
            cost = st.number_input("Maximum workload cost ($)", min_value=0.0, value=10.0, step=0.1)
            extra = {}
            for field in ("first_answer_s", "end_to_end_s"):
                if st.checkbox("Limit " + LABELS[field].lower(), key="enable_" + field):
                    extra[field] = st.number_input(LABELS[field] + " maximum", min_value=0.0, value=10.0, key="limit_" + field)
        st.subheader("Rank by")
        mode = st.radio("Ranking method", ["Single metric", "Weighted composite"], label_visibility="collapsed")
        weights = {}
        if mode == "Single metric":
            objective = st.selectbox("Optimize", list(LABELS), index=1,
                                     format_func=lambda k: LABELS[k] + (" ↑" if core.METRICS[k][1] else " ↓"), label_visibility="collapsed")
            objective = {"cost_usd": "cost", "ttft_s": "latency"}.get(objective, objective)
        else:
            objective = "composite"
            preset = st.selectbox("Weight preset", ["Balanced", "Quality first", "Budget first", "Speed first", "Custom"])
            presets = {"Balanced": (40,30,20,10), "Quality first": (70,10,10,10),
                       "Budget first": (20,60,10,10), "Speed first": (20,10,30,40), "Custom": (40,30,20,10)}
            defaults = dict(zip(("quality","cost_usd","ttft_s","output_tokens_s"), presets[preset]))
            with st.expander("Adjust weights"):
                for field in LABELS:
                    weights[field] = st.slider(LABELS[field], 0, 100, defaults.get(field, 0), key="weight_" + preset + field)
                st.caption("Relative weights. Zero ignores a metric.")
        return dict(slugs=slugs, input_tokens=int(inp), output_tokens=int(out), min_quality=quality,
                    max_ttft=ttft, max_cost=cost, objective=objective, weights=weights, extra_limits=extra)


def similarity_panel(snapshot, input_tokens, output_tokens):
    st.caption("Find alternatives across the full snapshot. Pick a model and the features that matter.")
    rows = core.comparison_rows(snapshot, [m["slug"] for m in snapshot["models"]], input_tokens, output_tokens)
    names = {r["slug"]: r["name"] for r in rows}
    reference_col, features_col = st.columns([1,2])
    target = reference_col.selectbox("Reference model", list(names), format_func=lambda s: f"{names[s]} · {s}",
                          key="similar_reference_" + snapshot["snapshot_id"][:12])
    features = features_col.multiselect("Similarity features", list(LABELS), default=["cost_usd", "ttft_s"], format_func=LABELS.get)
    minimum = st.slider("Minimum similarity (%)", 0, 100, 0,
                        help="0 shows all comparable models, ordered from closest to farthest. Raise it to narrow the results.")
    if not features:
        st.info("Choose at least one similarity feature.")
        return
    try:
        result = core.find_similar_models(rows, target, features, minimum)
    except ValueError as exc:
        st.info(str(exc))
        return
    st.caption("Reference · " + " · ".join(f"{LABELS[f]}: {result['reference'][f]:,.6g}" for f in features))
    st.caption(f"{len(result['matches'])} matches · {len(result['excluded'])} excluded for missing selected features · "
               f"{result['below_threshold']} below your similarity threshold. The reference itself is omitted.")
    if result["matches"]:
        display = [{"Model": r["name"], "ID": r["slug"], "Similarity (%)": r["similarity_pct"],
                    **{LABELS[f]: r[f] for f in features},
                    **{"Δ " + LABELS[f]: r["differences"][f] for f in features}} for r in result["matches"]]
        st.dataframe(pd.DataFrame(display), hide_index=True, width="stretch",
                     column_config={"Similarity (%)": st.column_config.NumberColumn(format="%.2f")})
        st.caption("All matches are in the scrollable table. Δ is candidate minus reference, in the original units.")
    else:
        st.info("No models match these features and this threshold. Lower the threshold or choose different features.")
    with st.expander("Similarity method and missing data"):
        st.write("For each chosen feature, divide the absolute difference from the reference by that feature's full snapshot range. "
                 "Combine these distances with an equally weighted root mean square. Similarity = 100 × (1 − distance). "
                 "Identical values score 100; constant features contribute zero distance. Close does not mean better. "
                 "Ranges include all known values in the snapshot; outliers can compress distances. Scores are specific to this snapshot and feature selection.")
        st.write("A candidate needs every selected feature; missing values are never treated as zero. "
                 "Workload cost uses the input/output token counts in the sidebar. Ranking limits do not filter this search.")
        st.json({"normalization_bounds": result["normalization_bounds"], "excluded": result["excluded"]})
    export = {**result, "snapshot_id": snapshot["snapshot_id"], "source": snapshot["source"],
              "retrieved_at": snapshot["retrieved_at"], "intelligence_index_version": snapshot["intelligence_index_version"],
              "is_demo": snapshot["is_demo"], "input_tokens": input_tokens, "output_tokens": output_tokens}
    json_download("Download similarity results", export, "similar_models.json")


def show_results(comparison, snapshot):
    winner, rows = comparison["winner"], comparison["candidates"]
    choice, cost, count = st.columns([2,1,1])
    choice.metric("Python's choice", winner["name"] if winner else "No feasible model")
    cost.metric("Estimated workload cost", f"${winner['cost_usd']:,.4f}" if winner else "—")
    count.metric("Candidates", len(rows))
    st.caption(f"{len(comparison['ranking'])} meet all requirements · {comparison['constraints']['input_tokens']:,} input / {comparison['constraints']['output_tokens']:,} output tokens")
    if not winner:
        st.info("No candidate meets every requirement. Review the limits or missing measurements.")
    if comparison["ranking"]:
        frame = pd.DataFrame(comparison["ranking"])
        objective = comparison["constraints"]["objective"]
        metric = "score" if objective == "composite" else {"cost":"cost_usd", "latency":"ttft_s"}.get(objective, objective)
        fields = list(dict.fromkeys(["name", "quality", "cost_usd", "ttft_s", metric]))
        st.dataframe(frame[fields].rename(columns={**LABELS, "name":"Model", "score":"Composite / 100"}),
                     hide_index=True, width="stretch")
    with st.expander("Charts & tradeoffs"):
        if comparison["ranking"]:
            chart = px.bar(frame.head(20), x=metric, y="slug", orientation="h", labels=LABELS,
                           color_discrete_sequence=["#cc0000"])
            chart.update_layout(yaxis={"autorange":"reversed"}, height=max(260, min(650, len(frame)*30+80)),
                                margin=dict(l=0,r=10,t=10,b=10), paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(chart, width="stretch")
        left, right = st.columns(2)
        x = left.selectbox("Horizontal metric", list(LABELS), index=1, format_func=LABELS.get)
        y = right.selectbox("Vertical metric", list(LABELS), format_func=LABELS.get)
        visible = pd.DataFrame(rows).dropna(subset=[x,y]).copy()
        if not visible.empty:
            frontier = core.pareto_front(rows, tuple(dict.fromkeys((x,y))))
            visible["Tradeoff"] = visible["slug"].map(lambda s: "Pareto frontier" if s in frontier else "Dominated")
            chart = px.scatter(visible, x=x, y=y, color="Tradeoff", symbol="eligible", hover_name="slug",
                               hover_data=["name"], labels=LABELS, color_discrete_map={"Pareto frontier":"#cc0000", "Dominated":"#999999"})
            chart.update_traces(marker=dict(size=11))
            st.plotly_chart(chart, width="stretch")
        st.caption("Frontier uses these two metrics across all selected models, including those that fail requirements. Missing chart values are omitted.")
    with st.expander("All measurements & exclusions"):
        full = pd.DataFrame(rows).copy()
        full["exclusions"] = full["exclusions"].apply(lambda x: "; ".join(x))
        st.dataframe(full.rename(columns=LABELS), hide_index=True, width="stretch")
    with st.expander("Scoring method"):
        st.write("Single metrics minimize prices/timing and maximize intelligence/speed. Ties use exact model ID.")
        st.write("Composite scoring first applies every requirement and removes models missing an active metric. "
                 "Each active metric is normalized to 0–1 across the remaining candidates, with lower-is-better metrics reversed. "
                 "Constant metrics give everyone 1. Score = 100 × sum(weight × utility) / sum(weights). "
                 "Changing candidates or limits changes the scale; scores are specific to this comparison.")
        st.write("Missing measurements are unknown, not zero. Benchmark quality is a proxy. Token counts recalculate cost, "
                 "not measured latency; include billed reasoning tokens. Prices exclude discounts and caching. "
                 "First-token, first-answer, and end-to-end timing are different clocks.")
        st.json({"weights": comparison["constraints"].get("weights"), "normalization_bounds": comparison["normalization_bounds"]})
        if winner:
            st.json({"winner_score_contributions": winner["contributions"]})
    with st.expander("Export results"):
        json_download("Comparison JSON", comparison, "comparison.json")
        exported = pd.DataFrame(rows).copy()
        if comparison["ranking"]:
            exported["score"] = exported["slug"].map({r["slug"]:r["score"] for r in comparison["ranking"]})
        exported["exclusions"] = exported["exclusions"].apply(lambda x: "; ".join(x))
        for col in exported.select_dtypes(include="object"):
            exported[col] = exported[col].map(lambda x: "'" + x if isinstance(x, str) and x.startswith(("=","+","-","@","\t","\r")) else x)
        st.download_button("Comparison CSV", exported.to_csv(index=False), "comparison.csv", "text/csv")


def agent_panel(snapshot, task, cloud=False):
    st.subheader("Agent")
    with st.expander("Controller settings"):
        hosted_only = cloud or str(settings("HOSTED_ONLY", "false")).lower() == "true"
        providers = ["Hosted API", "Scripted rehearsal (no LLM)"] if hosted_only else ["Local Qwen (notebook model)", "Ollama", "Hosted API", "Scripted rehearsal (no LLM)"]
        provider = st.selectbox("Controller", providers)
        key, model = "", "Qwen/Qwen2.5-1.5B-Instruct"
        if provider.startswith("Local"):
            st.caption("Uses Qwen2.5-1.5B-Instruct on this computer. First run downloads about 3 GB; CPU generation may be slow. Install the optional local-model requirements first.")
            with st.expander("Local model setup"):
                st.code('python -m pip install "transformers==4.57.6" "torch>=2.6,<3" "safetensors>=0.4,<1"', language="bash")
                st.caption("The Python app and scripted rehearsal do not need these packages. A GPU is optional. Model downloads and generation start only when you press Run.")
        elif provider == "Ollama":
            model = st.text_input("Installed Ollama model", value="qwen2.5:1.5b")
            st.caption("Start Ollama on the app's server and run: ollama pull qwen2.5:1.5b. A hosted app cannot reach a visitor's localhost.")
        elif provider == "Hosted API":
            model = st.text_input("Hosted model ID", value=settings("LLM_MODEL", ""))
            key = st.text_input("API key (this session only)", type="password") or settings("LLM_API_KEY", "")
            st.caption("Sends the task and selected comparison data to the configured provider. Configure LLM_BASE_URL on the server; keys are never included in exports.")
            if not settings("LLM_BASE_URL"):
                st.warning("No hosted provider configured. Set LLM_BASE_URL and LLM_MODEL in server environment or Streamlit secrets.")
        else:
            st.caption("Deterministic teaching walkthrough. This mode does not call an LLM.")
    fingerprint = context_id(snapshot, task)
    missing_hosted_setup = provider == "Hosted API" and (not settings("LLM_BASE_URL") or not model.strip())
    if missing_hosted_setup:
        st.info("The instructor must configure an LLM provider to enable the agent. Model comparison and similarity search are ready to use.")
    if st.button("Run checked agent", type="primary", key="run_agent", disabled=missing_hosted_setup):
        st.session_state.pop("agent_run", None)
        try:
            if provider.startswith("Local"):
                # Preflight avoids a cryptic failure for a missing optional installation.
                import importlib.util
                if not all(importlib.util.find_spec(p) for p in ("torch", "transformers", "safetensors")):
                    raise ValueError('Open Controller settings → Local model setup, install the listed packages, then restart the app.')
                controller = local_controller
            elif provider == "Ollama":
                controller = http_controller("ollama", model, base_url=settings("OLLAMA_HOST", "http://localhost:11434"))
            elif provider == "Hosted API":
                controller = http_controller("hosted", model, key, base_url=settings("LLM_BASE_URL"))
            else:
                controller = rehearsal_controller(snapshot, task)
            with st.spinner("Running controller and checking each action…"):
                result = run_agent(snapshot, task, controller)
            result["controller"] = {"provider": provider, "model": model if not provider.startswith("Scripted") else "none"}
            st.session_state["agent_run"] = result
        except ValueError as exc:
            st.error(str(exc))
    result = st.session_state.get("agent_run")
    if not result:
        return
    if result["context_id"] != fingerprint:
        st.warning("The snapshot or requirements changed. Run the agent again; the previous result is no longer shown as current.")
        return
    (st.success if result["completed"] else st.warning)(
        "Checked: required steps, exact winner, and numeric evidence match Python." if result["completed"] else "Agent did not pass verification: " + result["stop"])
    st.caption(f"Controller: {result['controller']['provider']} · {result['tool_calls']} tool attempts · {result['invalid_calls']} rejected calls")
    if result.get("recommendation"):
        st.write("Agent response · explanation text is not independently fact-checked")
        st.json(result["recommendation"])
    with st.expander("Inspect every tool request and result", expanded=not result["completed"]):
        st.json(result["trace"])
    json_download("Download agent audit", result, "agent_audit.json")


def learning_panel(baseline):
    with st.expander("The notebook's ten functions · code & execution trace"):
        functions = [core.has_measurement, core.read_metric, core.models_for_slugs, core.workload_cost,
                     core.meets_limits, core.choose_model, core.valid_tool_call, core.can_continue,
                     core.record_trace, core.check_recommendation]
        selected = st.selectbox("Inspect a function", range(10), format_func=lambda i: f"Q{i+1} · {functions[i].__name__}")
        st.code(inspect.getsource(functions[selected]), language="python")
        st.caption("Q6 preserves the original cost/latency choices. rank_models adds other objectives; the agent additionally checks numeric evidence. Tests and maintenance tools are in the developer repository: https://github.com/elaheJ/it344-model-lab-dev")
        st.write(f"Direct Python workflow · {baseline['tool_calls']} calls")
        st.json(baseline["trace"])


def main(agent=False, cloud=False):
    setup(agent)
    snapshot = snapshot_controls()
    workspace = st.radio("Workspace", ["Rank models", "Find similar models"], horizontal=True, label_visibility="collapsed")
    if workspace == "Find similar models":
        with st.sidebar:
            st.subheader("Workload")
            inp = st.number_input("Input tokens", min_value=0, max_value=10**12, value=100_000, step=10_000)
            out = st.number_input("Output tokens (including reasoning)", min_value=0, max_value=10**12, value=20_000, step=1_000)
        similarity_panel(snapshot, int(inp), int(out))
        return
    task = task_controls(snapshot, agent)
    if not task["slugs"]:
        st.info("Select at least one model in the sidebar to start comparing.")
        return
    try:
        baseline = core.fixed_workflow(snapshot, task)
    except ValueError as exc:
        st.warning(str(exc))
        return
    if snapshot["is_demo"]:
        st.caption("Fictional demo measurements · Load an AA snapshot to compare real models.")
    else:
        st.caption(f"SNAPSHOT · {snapshot['retrieved_at']} · Intelligence index version {snapshot['intelligence_index_version']}")
    if agent:
        agent_panel(snapshot, task, cloud=cloud)
        st.divider()
    show_results(baseline["comparison"], snapshot)
    learning_panel(baseline)
