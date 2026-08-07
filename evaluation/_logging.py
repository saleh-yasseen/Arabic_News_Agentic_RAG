"""Shared logging helpers for evaluation scripts.

Every eval run writes a JSON file to evaluation/logs/ with a filename of the
form eval_YYYYMMDD_HHMMSS_NNN.json so multiple runs in the same second don't
collide. The payload is intentionally flat: every run carries its eval type,
timestamp, aggregate results, and (when relevant) per-query details.
"""

import os
import json
from datetime import datetime

LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


def save_evaluation_run(eval_type, results, query_details=None):
    """Save an evaluation run to a numbered, timestamped file.

    Args:
        eval_type: short label written into the payload, e.g. "generation",
            "retrieval", "routing", "latency", "all".
        results: aggregate metrics dict produced by the eval run.
        query_details: optional per-query/per-item list (judge scores, latency
            breakdowns, retrieved ids, predicted tools, etc.).

    Returns:
        Absolute path of the saved file.
    """
    os.makedirs(LOGS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sequence = 1
    while True:
        filename = f"eval_{timestamp}_{sequence:03d}.json"
        filepath = os.path.join(LOGS_DIR, filename)
        if not os.path.exists(filepath):
            break
        sequence += 1

    save_data = {
        "eval_type": eval_type,
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "query_details": query_details or [],
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"[{eval_type}] Evaluation run saved to: {filepath}")
    return filepath
