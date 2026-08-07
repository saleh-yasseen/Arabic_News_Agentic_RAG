import time
import os
import json
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.graph import app as agent_app, call_llm_with_retry
from langchain_groq import ChatGroq
from _logging import save_evaluation_run

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "generation.json")

JUDGE_PROMPT = """أنت مقيّم محايد لجودة إجابات نظام أخبار عربي. بناءً على السؤال والسياق المسترجع والإجابة المولّدة، قيّم الإجابة على مقياس من 1 إلى 5 لكل معيار:

- correctness: هل الإجابة متسقة منطقياً وواقعياً مع السياق المعطى؟
- groundedness: هل كل ادعاء في الإجابة مدعوم فعلياً بالسياق، دون إضافة معلومات غير موجودة فيه؟
- completeness: هل غطت الإجابة الجوانب المهمة التي يسمح بها السياق المتاح؟

السؤال: {query}

السياق:
{context}

الإجابة:
{response}

أجب بصيغة JSON فقط، دون أي نص إضافي أو علامات markdown:
{{"correctness": <1-5>, "groundedness": <1-5>, "completeness": <1-5>}}
"""


def run_agent(query):
    state = {
        "query": query, "tool_choice": None, "context": "", "response": None,
        "loop_count": 0, "sources": [], "comparison": None,
        "model": "llama-3.3-70b-versatile", "tool_override": None,
        "retrieval_mode": "hybrid", "temperature": 0.0,
        "top_k": 5, "category_filter": None, "use_reranker": True, "min_score": None,
    }
    result = agent_app.invoke(state)
    return result.get("context", ""), result.get("response", "")


def judge(query, context, response):
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    prompt = JUDGE_PROMPT.format(query=query, context=context[:3000], response=response)
    raw = call_llm_with_retry(llm, prompt, max_retries=10, initial_delay=10).content.strip()
    cleaned = re.sub(r"^```json|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"  judge parse failed, raw output: {raw[:200]}")
        return {"correctness": None, "groundedness": None, "completeness": None}


def run_generation_eval():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    scores = {"correctness": [], "groundedness": [], "completeness": []}
    query_details = []  # To store details for logging

    for item in data:
        context, response = run_agent(item["query"])
        result = judge(item["query"], context, response)
        for key in scores:
            if result.get(key) is not None:
                scores[key].append(result[key])

        # Store query details for logging
        query_details.append({
            "query": item["query"],
            "context_length": len(context),
            "response_length": len(response),
            "judge_scores": result
        })

        time.sleep(10)

    averages = {
        key: round(sum(vals) / len(vals), 2) if vals else None
        for key, vals in scores.items()
    }

    results = {"n": len(data), "averages": averages}
    return results, query_details


def print_generation_report(results):
    print(f"Generation Evaluation — {results['n']} queries (LLM-as-judge, no reference answers)\n")
    for key, val in results["averages"].items():
        print(f"  {key:<14} {val}/5" if val is not None else f"  {key:<14} n/a")


if __name__ == "__main__":
    results, query_details = run_generation_eval()
    print_generation_report(results)
    save_evaluation_run("generation", results, query_details)
