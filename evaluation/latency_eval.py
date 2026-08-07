import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import time
from agent.graph import call_llm_with_retry
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.tools import _retrieve, _rerank, model as dense_model
from agent.graph import GENERATION_PROMPT
from langchain_groq import ChatGroq
from _logging import save_evaluation_run

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation", "data", "retrieval.json")


def measure_one(query):
    t0 = time.perf_counter()
    dense_model.encode(query, normalize_embeddings=True)
    t1 = time.perf_counter()

    fused = _retrieve(query, top_k=15, mode="hybrid")
    t2 = time.perf_counter()

    reranked, _ = _rerank(query, fused, top_k=5)
    t3 = time.perf_counter()

    context = " ".join([p.payload["text"] for p in reranked])
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    prompt = GENERATION_PROMPT.format(context=context, query=query)
    call_llm_with_retry(llm, prompt, max_retries=5, initial_delay=10)
    t4 = time.perf_counter()
    time.sleep(2)

    return {
        "embedding_ms": (t1 - t0) * 1000,
        "retrieval_ms": (t2 - t1) * 1000,
        "reranker_ms": (t3 - t2) * 1000,
        "generation_ms": (t4 - t3) * 1000,
        "total_ms": (t4 - t0) * 1000,
    }


def run_latency_eval():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries = [d["query"] for d in data]
    runs = [measure_one(q) for q in queries]

    keys = ["embedding_ms", "retrieval_ms", "reranker_ms", "generation_ms", "total_ms"]
    averages = {k: round(sum(r[k] for r in runs) / len(runs), 1) for k in keys}
    query_details = [
        {"query": q, **run} for q, run in zip(queries, runs)
    ]
    return {"n": len(queries), "averages": averages}, query_details


def print_latency_report(results):
    a = results["averages"]
    print(f"Latency Evaluation — averaged over {results['n']} queries\n")
    print(f"  Embedding   {a['embedding_ms']:>8} ms")
    print(f"  Retrieval   {a['retrieval_ms']:>8} ms")
    print(f"  Reranker    {a['reranker_ms']:>8} ms")
    print(f"  Generation  {a['generation_ms']:>8} ms")
    print(f"  {'Total':<11} {a['total_ms']/1000:>7.2f} sec")


if __name__ == "__main__":
    results, query_details = run_latency_eval()
    print_latency_report(results)
    save_evaluation_run("latency", results, query_details)
