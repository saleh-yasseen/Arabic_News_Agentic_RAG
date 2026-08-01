import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.graph import route_query

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "routing.json")
TOOLS = ["search_news", "summarize_topic", "compare_timeline", "answer_direct"]


def load_data():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def predict_tool(query):
    state = {"query": query, "tool_override": None, "model": "llama-3.3-70b-versatile"}
    return route_query(state)["tool_choice"]


def run_routing_eval():
    data = load_data()
    confusion = {t: {t2: 0 for t2 in TOOLS} for t in TOOLS}
    correct = 0

    for item in data:
        predicted = predict_tool(item["query"])
        expected = item["expected_tool"]
        if predicted not in confusion.get(expected, {}):
            confusion.setdefault(expected, {}).setdefault(predicted, 0)
        confusion[expected][predicted] = confusion[expected].get(predicted, 0) + 1
        if predicted == expected:
            correct += 1

    overall_accuracy = round(correct / len(data), 4)

    per_tool_accuracy = {}
    for tool in TOOLS:
        tool_items = [d for d in data if d["expected_tool"] == tool]
        if not tool_items:
            continue
        tool_correct = confusion[tool].get(tool, 0)
        per_tool_accuracy[tool] = round(tool_correct / len(tool_items), 4)

    return {
        "overall_accuracy": overall_accuracy,
        "per_tool_accuracy": per_tool_accuracy,
        "confusion_matrix": confusion,
        "n": len(data),
    }


def print_routing_report(results):
    print("Router Evaluation\n")
    for tool, acc in results["per_tool_accuracy"].items():
        print(f"  {tool:<18} {acc*100:.1f}%")
    print(f"\n  {'Overall':<18} {results['overall_accuracy']*100:.1f}%")

    print("\nConfusion matrix (rows = expected, cols = predicted)")
    header = "".join(f"{t[:10]:>12}" for t in TOOLS)
    print(f"{'':<18}{header}")
    for expected in TOOLS:
        row = "".join(f"{results['confusion_matrix'][expected].get(p, 0):>12}" for p in TOOLS)
        print(f"{expected:<18}{row}")


if __name__ == "__main__":
    results = run_routing_eval()
    print_routing_report(results)
