import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.tools import _retrieve, _retrieve_dense_only, _retrieve_sparse_only, _rerank
from _logging import save_evaluation_run

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "retrieval.json")


def load_data():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_data(data):
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def recall_at_k(retrieved_ids, relevant_ids, k):
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & set(relevant_ids)) / len(relevant_ids)


def precision_at_k(retrieved_ids, relevant_ids, k):
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    return len(set(top_k) & set(relevant_ids)) / len(top_k)


def mrr(retrieved_ids, relevant_ids):
    for i, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0


def label_dataset():
    data = load_data()
    for item in data:
        print(f"\nQuery: {item['query']}")
        if item["relevant_docs"]:
            print(f"  already labeled ({len(item['relevant_docs'])} relevant) — press Enter to skip, or 'r' to relabel")
            choice = input("> ").strip()
            if choice.lower() != "r":
                continue

        candidates = _retrieve(item["query"], top_k=15, mode="hybrid")
        for i, c in enumerate(candidates, 1):
            snippet = c.payload["text"][:100].replace("\n", " ")
            print(f"  {i:>2}. [{c.payload['category']}] {snippet}...")

        raw = input("Relevant indices (comma-separated, or 'none'): ").strip()
        if raw.lower() == "none" or not raw:
            item["relevant_docs"] = []
        else:
            indices = [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]
            item["relevant_docs"] = [str(candidates[i - 1].id) for i in indices if 1 <= i <= len(candidates)]
        save_data(data)
        print(f"  saved {len(item['relevant_docs'])} relevant docs")

    print("\nLabeling complete.")


def run_retrieval_eval(k5=5, k10=10):
    data = load_data()
    labeled = [d for d in data if d["relevant_docs"]]
    if not labeled:
        print("No labeled queries found. Run with --label first.")
        return None, []

    def dense_ids(q, k):
        return [str(p.id) for p in _retrieve_dense_only(q, top_k=k)]

    def sparse_ids(q, k):
        return [str(p.id) for p in _retrieve_sparse_only(q, top_k=k)]

    def hybrid_ids(q, k):
        return [str(p.id) for p in _retrieve(q, top_k=k, mode="hybrid")]

    def hybrid_reranked_ids(q, k):
        fused = _retrieve(q, top_k=max(15, k * 3), mode="hybrid")
        reranked, _ = _rerank(q, fused, top_k=k)
        return [str(p.id) for p in reranked]

    modes = {
        "Dense": dense_ids,
        "Sparse": sparse_ids,
        "Hybrid": hybrid_ids,
        "Hybrid + Reranker": hybrid_reranked_ids,
    }

    results = {}
    query_details = []
    for mode_name, fn in modes.items():
        r5s, r10s, p5s, rrs = [], [], [], []
        for item in labeled:
            relevant = item["relevant_docs"]
            ids_10 = fn(item["query"], k10)
            r5 = recall_at_k(ids_10, relevant, k5)
            r10 = recall_at_k(ids_10, relevant, k10)
            p5 = precision_at_k(ids_10, relevant, k5)
            rr = mrr(ids_10, relevant)
            r5s.append(r5)
            r10s.append(r10)
            p5s.append(p5)
            rrs.append(rr)
            query_details.append({
                "mode": mode_name,
                "query": item["query"],
                "retrieved_ids": ids_10,
                "relevant_ids": relevant,
                "recall_at_5": r5,
                "recall_at_10": r10,
                "precision_at_5": p5,
                "mrr": rr,
            })
            time.sleep(5)

        n = len(labeled)
        results[mode_name] = {
            "recall_at_5": round(sum(r5s) / n, 3),
            "recall_at_10": round(sum(r10s) / n, 3),
            "precision_at_5": round(sum(p5s) / n, 3),
            "mrr": round(sum(rrs) / n, 3),
        }

    return {"n_labeled": len(labeled), "modes": results}, query_details


def print_retrieval_report(results):
    if not results:
        return
    print(f"Retrieval Evaluation — {results['n_labeled']} labeled queries\n")
    print(f"{'Method':<20}{'Recall@5':>10}{'Recall@10':>11}{'Precision@5':>13}{'MRR':>8}")
    for mode, m in results["modes"].items():
        print(f"{mode:<20}{m['recall_at_5']:>10}{m['recall_at_10']:>11}{m['precision_at_5']:>13}{m['mrr']:>8}")


if __name__ == "__main__":
    if "--label" in sys.argv:
        label_dataset()
    else:
        results, query_details = run_retrieval_eval()
        print_retrieval_report(results)
        if results is not None:
            save_evaluation_run("retrieval", results, query_details)
