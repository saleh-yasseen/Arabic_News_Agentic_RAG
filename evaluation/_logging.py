"""Shared logging helpers for evaluation scripts.

Each run lands in its own subfolder under evaluation/logs/:

    logs/eval_YYYYMMDD_HHMMSS_NNN/
        routing.json            # eval_type="routing" payload
        retrieval.json          # eval_type="retrieval" payload
        generation.json         # eval_type="generation" payload
        latency.json            # eval_type="latency" payload
        all.json                # combined run payload (run_eval.py only)
        summary.json            # headline metrics across all sections
        summary.md              # human-readable counterpart

Subsequent calls to save_evaluation_run() within the same process reuse the
same run folder, so a combined run_eval invocation produces a single folder
with one file per section plus one summary.
"""

import os
import json
from datetime import datetime

LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

# Module-level cache so a single run_eval.py invocation reuses one folder.
_current_run_id = None
_current_run_dir = None
_current_run_timestamp = None


def _next_sequence(timestamp):
    """Find the next free 3-digit sequence number for this timestamp."""
    sequence = 1
    while True:
        candidate = f"eval_{timestamp}_{sequence:03d}"
        if not os.path.exists(os.path.join(LOGS_DIR, candidate)):
            return sequence
        sequence += 1


def _start_new_run():
    """Allocate a new run_id and create its folder."""
    global _current_run_id, _current_run_dir, _current_run_timestamp
    _current_run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seq = _next_sequence(_current_run_timestamp)
    _current_run_id = f"eval_{_current_run_timestamp}_{seq:03d}"
    _current_run_dir = os.path.join(LOGS_DIR, _current_run_id)
    os.makedirs(_current_run_dir, exist_ok=True)
    return _current_run_id, _current_run_dir


def _ensure_run(run_id=None):
    """Return (run_id, run_dir), starting a new run if needed."""
    global _current_run_id, _current_run_dir
    if run_id:
        _current_run_id = run_id
        _current_run_dir = os.path.join(LOGS_DIR, run_id)
        os.makedirs(_current_run_dir, exist_ok=True)
        return _current_run_id, _current_run_dir
    if _current_run_id is None:
        return _start_new_run()
    return _current_run_id, _current_run_dir


def save_evaluation_run(eval_type, results, query_details=None, run_id=None):
    """Persist one eval section into the current run folder.

    Args:
        eval_type: short label — "routing", "retrieval", "generation",
            "latency", or "all".
        results: aggregate metrics dict produced by the eval run.
        query_details: optional per-query/per-item list.
        run_id: optional explicit run_id (overrides the in-process cache).

    Returns:
        Dict with keys: run_id, folder, data. ``folder`` is the run directory,
        ``data`` is the path of the per-section JSON file just written.
    """
    rid, folder = _ensure_run(run_id)

    payload = {
        "eval_type": eval_type,
        "run_id": rid,
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "query_details": query_details or [],
    }

    data_path = os.path.join(folder, f"{eval_type}.json")
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[{eval_type}] Saved → {data_path}")
    return {"run_id": rid, "folder": folder, "data": data_path}


def current_run_folder():
    """Return (run_id, folder) for the active run, allocating one if needed."""
    return _ensure_run()


# ---------------------------------------------------------------------------
# Summary writers
# ---------------------------------------------------------------------------

SUMMARY_SECTIONS = ("routing", "retrieval", "generation", "latency")


def _empty_summary_section():
    return {"present": False}


def _routing_section(results):
    if not results:
        return _empty_summary_section()
    per_tool = results.get("per_tool_accuracy", {}) or {}
    return {
        "present": True,
        "overall_accuracy_pct": round(results.get("overall_accuracy", 0) * 100, 1),
        "per_tool_accuracy_pct": {
            tool: round(acc * 100, 1) for tool, acc in per_tool.items()
        },
        "n_queries": results.get("n", 0),
    }


def _retrieval_section(results):
    if not results:
        return _empty_summary_section()
    modes = results.get("modes", {}) or {}
    if not modes:
        return _empty_summary_section()
    best_mode = max(modes.items(), key=lambda x: x[1].get("recall_at_5", 0))[0]
    best = modes[best_mode]
    return {
        "present": True,
        "n_labeled": results.get("n_labeled", 0),
        "best_mode": best_mode,
        "recall_at_5_pct": round(best.get("recall_at_5", 0) * 100, 1),
        "recall_at_10_pct": round(best.get("recall_at_10", 0) * 100, 1),
        "precision_at_5_pct": round(best.get("precision_at_5", 0) * 100, 1),
        "mrr_pct": round(best.get("mrr", 0) * 100, 1),
        "per_mode": {
            name: {k: round(v.get(k, 0) * 100, 1) for k in
                   ("recall_at_5", "recall_at_10", "precision_at_5", "mrr")}
            for name, v in modes.items()
        },
    }


def _generation_section(results):
    if not results:
        return _empty_summary_section()
    avgs = results.get("averages", {}) or {}
    return {
        "present": True,
        "n": results.get("n", 0),
        "correctness_x_of_5": avgs.get("correctness"),
        "groundedness_x_of_5": avgs.get("groundedness"),
        "completeness_x_of_5": avgs.get("completeness"),
    }


def _latency_section(results):
    if not results:
        return _empty_summary_section()
    avgs = results.get("averages", {}) or {}
    total_ms = avgs.get("total_ms", 0) or 0
    return {
        "present": True,
        "n": results.get("n", 0),
        "embedding_ms": avgs.get("embedding_ms"),
        "retrieval_ms": avgs.get("retrieval_ms"),
        "reranker_ms": avgs.get("reranker_ms"),
        "generation_ms": avgs.get("generation_ms"),
        "total_sec": round(total_ms / 1000.0, 2),
    }


_SECTION_BUILDERS = {
    "routing": _routing_section,
    "retrieval": _retrieval_section,
    "generation": _generation_section,
    "latency": _latency_section,
}


def build_summary(sections, run_id=None, timestamp=None):
    """Build the summary dict from a {eval_type: results} mapping."""
    if run_id is None:
        run_id = _current_run_id
    if timestamp is None:
        timestamp = datetime.now().isoformat()

    n_total = 0
    out = {"run_id": run_id, "timestamp": timestamp}
    for section_name in SUMMARY_SECTIONS:
        builder = _SECTION_BUILDERS[section_name]
        results = sections.get(section_name)
        section = builder(results)
        out[section_name] = section
        if section.get("present"):
            n_key = "n_queries" if section_name == "routing" else "n"
            n_total += section.get(n_key, 0) or 0

    out["n_total_queries"] = n_total
    return out


def _fmt_pct(x):
    return f"{x:.1f}%" if x is not None else "n/a"


def _fmt_x_of_5(x):
    return f"{x}/5" if x is not None else "n/a"


def render_summary_markdown(summary):
    lines = [
        f"# Evaluation Run — {summary['run_id']}",
        f"**Timestamp:** {summary['timestamp']}",
        "",
    ]

    routing = summary.get("routing", {})
    if routing.get("present"):
        lines.append("## Router")
        lines.append(f"- Overall: **{routing['overall_accuracy_pct']:.1f}%**")
        per_tool = routing.get("per_tool_accuracy_pct", {})
        if per_tool:
            chips = " | ".join(f"{t}: {acc:.1f}%" for t, acc in per_tool.items())
            lines.append(f"- {chips}")
        lines.append("")

    retrieval = summary.get("retrieval", {})
    if retrieval.get("present"):
        lines.append(f"## Retrieval (n_labeled = {retrieval.get('n_labeled', 0)})")
        per_mode = retrieval.get("per_mode", {})
        if per_mode:
            lines.append("")
            lines.append("| Method | Recall@5 | Recall@10 | Precision@5 | MRR |")
            lines.append("| --- | --- | --- | --- | --- |")
            best = retrieval.get("best_mode")
            for name, m in per_mode.items():
                row = (
                    f"| {name} | {_fmt_pct(m['recall_at_5'])} | "
                    f"{_fmt_pct(m['recall_at_10'])} | "
                    f"{_fmt_pct(m['precision_at_5'])} | "
                    f"{_fmt_pct(m['mrr'])} |"
                )
                if name == best:
                    row = row.replace(name, f"**{name}**", 1)
                lines.append(row)
        lines.append("")

    generation = summary.get("generation", {})
    if generation.get("present"):
        lines.append(f"## Generation (n = {generation.get('n', 0)})")
        lines.append(f"- Correctness: {_fmt_x_of_5(generation.get('correctness_x_of_5'))}")
        lines.append(f"- Groundedness: {_fmt_x_of_5(generation.get('groundedness_x_of_5'))}")
        lines.append(f"- Completeness: {_fmt_x_of_5(generation.get('completeness_x_of_5'))}")
        lines.append("")

    latency = summary.get("latency", {})
    if latency.get("present"):
        lines.append(f"## Latency (n = {latency.get('n', 0)})")
        lines.append(f"- Embedding: {latency.get('embedding_ms')} ms")
        lines.append(f"- Retrieval: {latency.get('retrieval_ms')} ms")
        lines.append(f"- Reranker: {latency.get('reranker_ms')} ms")
        lines.append(f"- Generation: {latency.get('generation_ms')} ms")
        lines.append(f"- **Total: {latency.get('total_sec')} sec**")
        lines.append("")

    return "\n".join(lines)


def write_summary(folder, sections, run_id=None, timestamp=None):
    """Write summary.json and summary.md for a run folder."""
    summary = build_summary(sections, run_id=run_id, timestamp=timestamp)

    json_path = os.path.join(folder, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    md_path = os.path.join(folder, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_summary_markdown(summary))

    print(f"[summary] Saved → {json_path}")
    print(f"[summary] Saved → {md_path}")
    return {"summary_json": json_path, "summary_md": md_path}
