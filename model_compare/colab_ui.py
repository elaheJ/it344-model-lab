"""Gradio classroom apps. Credentials are used only before the UI starts."""
import getpass
import inspect
import json
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.express as px

from . import core
from .agent import local_controller, rehearsal_controller, run_agent
from .snapshots import fetch_artificial_analysis

LABELS = {key: value[0] for key, value in core.METRICS.items()}
FIELDS = list(LABELS)
FUNCTIONS = [core.has_measurement, core.read_metric, core.models_for_slugs,
             core.workload_cost, core.meets_limits, core.choose_model,
             core.valid_tool_call, core.can_continue, core.record_trace,
             core.check_recommendation]
CSS = """
.gradio-container {max-width:1200px!important; background:#fff; color:#111;}
.eyebrow {color:#cc0000; font-size:12px; font-weight:700; letter-spacing:2px;}
h1,h2,h3 {letter-spacing:-.03em;}
"""


def prompt_snapshot():
    """Never save the key to a file, environment variable, UI state, or output."""
    key = getpass.getpass("Artificial Analysis API key (AA_API_KEY): ").strip()
    try:
        snapshot = fetch_artificial_analysis(key)
    except Exception:
        # Suppress request objects, response bodies and notebook traceback locals.
        print("Could not load AA data. Check your key, API access/quota and connection, then rerun this cell.")
        return None
    finally:
        key = ""
    print(f"Loaded {len(snapshot['models'])} models · {snapshot['retrieved_at']}")
    return snapshot


def build_task(slugs, inp, out, quality, ttft, cost, objective, weights, first=None, end=None):
    def tokens(value):
        if not core.has_measurement(value) or int(value) != value:
            raise ValueError("Token volumes must be whole numbers.")
        return int(value)
    task = dict(slugs=list(slugs or []), input_tokens=tokens(inp), output_tokens=tokens(out),
                min_quality=quality, max_ttft=ttft, max_cost=cost, objective=objective,
                weights=dict(zip(FIELDS, weights)), extra_limits={})
    for field, value in (("first_answer_s", first), ("end_to_end_s", end)):
        if value is not None:
            task["extra_limits"][field] = value
    if not core.valid_tool_call("compare_models", task):
        raise ValueError("Select models, nonnegative requirements, and a positive weight for composite ranking.")
    return task


def table(rows, similarity=False):
    columns = ["name", "slug", "creator", "similarity_pct" if similarity else "score", *FIELDS]
    records = []
    for row in rows:
        item = {key: row.get(key) for key in columns}
        if similarity:
            item.update({"Δ " + LABELS[key]: value for key, value in row["differences"].items()})
        records.append(item)
    return pd.DataFrame(records, columns=list(records[0]) if records else columns).rename(
        columns={**LABELS, "name": "Model", "slug": "Model ID", "creator": "Creator",
                 "score": "Score / 100", "similarity_pct": "Similarity %"})


def export_files(frame, result, prefix):
    directory = Path(tempfile.mkdtemp(prefix="model-lab-"))
    csv_path, json_path = directory / f"{prefix}.csv", directory / f"{prefix}.json"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return str(csv_path), str(json_path)


def comparison_outputs(snapshot, task, controller_kind=None):
    """One click uses one immutable task for both the baseline and the agent."""
    if controller_kind and len(task["slugs"]) > 12:
        raise ValueError("Select at most 12 models for the agent; regular comparison supports the full catalog.")
    result = core.fixed_workflow(snapshot, task)
    comparison, winner = result["comparison"], result["winner"]
    frame = table(comparison["ranking"])
    title = (f"Selected: {winner['name']} · workload ${winner['cost_usd']:.4f} · first token {winner['ttft_s']:g}s"
             if winner else "No model meets these requirements. Adjust the limits or choose other models.")
    title += f"\n\n{len(comparison['ranking'])} of {len(task['slugs'])} models qualify."
    points = pd.DataFrame(comparison["ranking"], columns=["name", "cost_usd", "quality", "ttft_s"])
    figure = px.scatter(points, x="cost_usd", y="quality", hover_name="name", color="ttft_s",
                        labels=LABELS, color_continuous_scale=["#111111", "#cc0000"])
    figure.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=20, b=10))
    if controller_kind:
        controller = (rehearsal_controller(snapshot, task) if controller_kind == "rehearsal" else local_controller)
        agent = run_agent(snapshot, task, controller)
        result["agent"] = agent
        if controller_kind == "rehearsal":
            title += "\n\nScripted rehearsal · no LLM was called."
        if agent["completed"]:
            title += "\n\nAgent completed: model choice and numeric evidence passed Python checks."
        else:
            title += "\n\nAgent did not complete verification (" + agent["stop"] + "). The table shows the direct Python result."
        title += " Free-text explanations are unverified; inspect the trace below."
    csv_file, json_file = export_files(frame, result, "comparison")
    return title, frame, figure, result, csv_file, json_file


def similarity_outputs(snapshot, slug, features, minimum, inp, out):
    # Reuse validation without imposing ranking eligibility on similarity search.
    task = build_task([slug], inp, out, 0, 1e15, 1e15, "cost", [0] * len(FIELDS))
    rows = core.comparison_rows(snapshot, [m["slug"] for m in snapshot["models"]],
                                task["input_tokens"], task["output_tokens"])
    result = core.find_similar_models(rows, slug, features, minimum)
    result.update(snapshot_id=snapshot["snapshot_id"], retrieved_at=snapshot["retrieved_at"],
                  source=snapshot["source"], input_tokens=task["input_tokens"], output_tokens=task["output_tokens"])
    frame = table(result["matches"], similarity=True)
    message = (f"{len(result['matches'])} matches · {len(result['excluded'])} missing selected features · "
               f"{result['below_threshold']} below your threshold. All matches are shown. "
               "Δ = alternative − reference; similarity measures closeness, not quality.")
    csv_file, json_file = export_files(frame, result, "similarity")
    reference = table([{**result["reference"], "similarity_pct": 100, "differences": {}}], similarity=True)
    return message, reference, frame, result, csv_file, json_file


def build_app(snapshot, agent=False):
    if snapshot is None:
        raise ValueError("Run the AA API key cell successfully before starting the app.")
    choices = [(f"{m['name']} · {m['slug']}", m["slug"]) for m in snapshot["models"]]
    rows = core.comparison_rows(snapshot, [m["slug"] for m in snapshot["models"]], 100_000, 20_000)
    complete = [r for r in rows if all(core.has_measurement(r.get(f)) for f in ("quality", "cost_usd", "ttft_s"))]
    initial = [r for r in complete if core.meets_limits(r, 30, 10, 10)] or complete
    defaults = [r["slug"] for r in sorted(initial, key=lambda r: (-r["quality"], r["slug"]))[:8]]
    defaults = defaults or [choices[0][1]]
    snapshot_file = Path(tempfile.mkdtemp(prefix="model-lab-")) / "snapshot.json"
    snapshot_file.write_text(json.dumps(snapshot, indent=2, allow_nan=False))

    with gr.Blocks(title="Model Lab" + (" · Agent" if agent else ""), analytics_enabled=False) as app:
        gr.HTML('<div class="eyebrow">ILLINOIS STATE UNIVERSITY / IT344</div>')
        gr.Markdown("# Model Lab" + (" · Agent" if agent else "") + "\nCompare models. Find alternatives for your project.")
        gr.Markdown(f"**AA snapshot:** {len(choices)} models · {snapshot['retrieved_at']} · index {snapshot['intelligence_index_version']}")
        with gr.Accordion("Project workload", open=False):
            with gr.Row():
                inp = gr.Number(value=100_000, minimum=0, maximum=10**12, precision=0, label="Total input tokens")
                out = gr.Number(value=20_000, minimum=0, maximum=10**12, precision=0, label="Total output tokens (including reasoning)")
            gr.Markdown("Example: 1,000 requests × 100 input tokens = 100,000 input tokens. Costs are estimates for these totals.")

        with gr.Tab("Compare"):
            models = gr.Dropdown(choices, value=defaults, multiselect=True, label="Models to compare", filterable=True)
            with gr.Accordion("Project requirements", open=False):
                with gr.Row():
                    quality = gr.Number(value=30, minimum=0, label="Minimum intelligence index")
                    ttft = gr.Number(value=10, minimum=0, label="Maximum first-token delay (s)")
                    cost = gr.Number(value=10, minimum=0, label="Maximum workload cost ($)")
                with gr.Row():
                    first = gr.Number(value=None, minimum=0, label="Maximum first-answer time (s; blank = no limit)")
                    end = gr.Number(value=None, minimum=0, label="Maximum end-to-end time (s; blank = no limit)")
            objective = gr.Dropdown([(label + (" ↑" if core.METRICS[key][1] else " ↓"), key) for key, label in LABELS.items()]
                                    + [("Weighted composite", "composite")], value="cost_usd", label="Rank by")
            with gr.Accordion("Composite weights", open=False):
                gr.Markdown("Choose **Weighted composite** above to use these relative weights. Zero ignores a metric.")
                weights = [gr.Slider(0, 100, value={"quality":40, "cost_usd":30, "ttft_s":20, "output_tokens_s":10}.get(key, 0),
                                     step=1, label=label) for key, label in LABELS.items()]
            inputs = [models, inp, out, quality, ttft, cost, objective, *weights, first, end]
            with gr.Row():
                compare = gr.Button("Compare models", variant="primary")
                if agent:
                    agent_button = gr.Button("Run checked agent", variant="primary")
            if agent:
                mode = gr.Radio([("Qwen in this runtime", "local"), ("Scripted rehearsal (no LLM)", "rehearsal")],
                                value="local", label="Agent controller")
                gr.Markdown("Select up to 12 candidates for the agent. Python verifies its tool calls, choice, and numeric evidence.")
            status = gr.Markdown("Set your requirements, then compare. Results describe the settings at the time you click.")
            ranking = gr.Dataframe(interactive=False, label="Ranking")
            chart = gr.Plot(label="Cost / intelligence / first-token delay")
            with gr.Row():
                csv_download = gr.DownloadButton("Download ranking CSV")
                json_download = gr.DownloadButton("Download result + trace JSON")
            with gr.Accordion("Evidence, exclusions and tool trace", open=False):
                trace = gr.JSON()
            outputs = [status, ranking, chart, trace, csv_download, json_download]

            def compare_click(*values):
                try:
                    task = build_task(*values[:7], values[7:15], *values[15:17])
                    return comparison_outputs(snapshot, task)
                except ValueError as exc:
                    return str(exc), pd.DataFrame(), None, {}, None, None
            compare.click(compare_click, inputs, outputs, api_name=False)
            if agent:
                def agent_click(*values):
                    try:
                        task = build_task(*values[:7], values[7:15], *values[15:17])
                        return comparison_outputs(snapshot, task, values[-1])
                    except ValueError as exc:
                        return str(exc), pd.DataFrame(), None, {}, None, None
                agent_button.click(agent_click, [*inputs, mode], outputs, api_name=False)

        with gr.Tab("Find similar models"):
            gr.Markdown("Search the **entire snapshot**, independent of the comparison shortlist and requirements.")
            reference = gr.Dropdown(choices, value=defaults[0], label="Reference model", filterable=True)
            features = gr.Dropdown([(label, key) for key, label in LABELS.items()], value=["cost_usd", "ttft_s"],
                                   multiselect=True, label="Similar in…")
            minimum = gr.Slider(0, 100, value=0, step=1, label="Minimum similarity (%) · 0 shows all comparable models")
            find = gr.Button("Find alternatives", variant="primary")
            similarity_status = gr.Markdown()
            reference_table = gr.Dataframe(interactive=False, label="Reference measurements")
            matches = gr.Dataframe(interactive=False, label="All matching alternatives")
            with gr.Row():
                similar_csv = gr.DownloadButton("Download alternatives CSV")
                similar_json = gr.DownloadButton("Download similarity details JSON")
            with gr.Accordion("How similarity is calculated", open=False):
                gr.Markdown("Each difference is divided by that feature's range in the full snapshot. "
                            "The equal-weight RMS distance becomes a 0–100 similarity score. "
                            "Missing selected measurements exclude a model; outliers can compress distances.")
                details = gr.JSON()

            def similar_click(*values):
                try:
                    return similarity_outputs(snapshot, *values)
                except ValueError as exc:
                    return str(exc), pd.DataFrame(), pd.DataFrame(), {}, None, None
            find.click(similar_click, [reference, features, minimum, inp, out],
                       [similarity_status, reference_table, matches, details, similar_csv, similar_json], api_name=False)

        with gr.Accordion("Learn from the notebook's ten functions", open=False):
            function = gr.Dropdown([(f"Q{i+1} · {fn.__name__}", i) for i, fn in enumerate(FUNCTIONS)], value=0, label="Function")
            code = gr.Code(inspect.getsource(FUNCTIONS[0]), language="python", interactive=False)
            function.change(lambda i: inspect.getsource(FUNCTIONS[int(i)]), function, code, api_name=False)
            gr.Markdown("Try changing a cost limit or a composite weight. Explain why the selected model changes. "
                        "Then find alternatives similar in cost and TTFT, and test your shortlist on real project examples.")
        gr.DownloadButton("Download this AA snapshot", value=str(snapshot_file))
        gr.Markdown("Data: [Artificial Analysis](https://artificialanalysis.ai). Benchmarks guide a shortlist; they do not replace project evaluation. "
                    "To refresh data, rerun the API key cell and then relaunch the app.")
    return app


def launch_app(snapshot, agent=False, share=True):
    """Colab uses a temporary Gradio link; the key is never passed to Gradio."""
    app = build_app(snapshot, agent)
    app.queue(default_concurrency_limit=1, max_size=8)
    app.launch(share=share, inline=True, show_error=False, css=CSS,
               theme=gr.themes.Base(primary_hue="red", neutral_hue="gray").set(
                   button_primary_background_fill="#cc0000", button_primary_background_fill_hover="#a30000",
                   button_primary_text_color="#ffffff", body_background_fill="#ffffff"))
    return app
