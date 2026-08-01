import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.tools import client, collection_name
from qdrant_client import models as qmodels
from langchain_groq import ChatGroq

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

CATEGORIES = ["Politics", "Finance", "Medical", "Religion", "Sports", "Tech", "Culture"]

TOOL_DESCRIPTIONS = {
    "search_news": "سؤال محدد وواقعي عن حدث أو موضوع إخباري معين، يتوقع إجابة مباشرة مبنية على خبر بعينه",
    "summarize_topic": "طلب نظرة عامة أو ملخص واسع عن موضوع، وليس سؤالاً عن تفصيل واحد محدد",
    "compare_timeline": "سؤال عن كيف تطور أو تغير موضوع ما عبر الزمن، أو مقارنة بين فترتين أو حدثين",
    "answer_direct": "سؤال معرفة عامة لا علاقة له بالأخبار، مثل تعريف مصطلح أو حقيقة عامة",
}

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.9)


def sample_chunks(per_category=3):
    """Stratified sample across known categories — plain random offset risks
    pulling only from whichever category was indexed first."""
    sampled = []
    for cat in CATEGORIES:
        points, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=qmodels.Filter(
                must=[qmodels.FieldCondition(key="category", match=qmodels.MatchValue(value=cat))]
            ),
            limit=50,
            with_payload=True,
        )
        if not points:
            continue
        take = min(per_category, len(points))
        sampled.extend(random.sample(points, take))
    return sampled


def generate_query_for_chunk(chunk_text):
    prompt = f"""اقرأ المقطع الإخباري التالي، ثم اكتب سؤالاً واحداً طبيعياً بالعربية يمكن لمستخدم حقيقي أن يطرحه، بحيث تكون الإجابة عليه موجودة مباشرة في هذا المقطع تحديداً. لا تكتب أي شيء غير السؤال نفسه.

المقطع:
{chunk_text[:600]}

السؤال:"""
    result = llm.invoke(prompt)
    return result.content.strip().strip('"').strip('"')


def generate_retrieval_and_generation_data(per_category=3):
    chunks = sample_chunks(per_category)
    retrieval_data = []
    generation_data = []

    for point in chunks:
        text = point.payload["text"]
        if len(text) < 100:
            continue
        query = generate_query_for_chunk(text)
        retrieval_data.append({
            "query": query,
            "relevant_docs": [str(point.id)],
            "note": "synthetic — one known-relevant chunk only; treat recall as a lower bound, since a genuinely relevant near-duplicate chunk elsewhere would be scored as a miss"
        })
        generation_data.append({"query": query})

    with open(os.path.join(DATA_DIR, "retrieval.json"), "w", encoding="utf-8") as f:
        json.dump(retrieval_data, f, ensure_ascii=False, indent=2)
    with open(os.path.join(DATA_DIR, "generation.json"), "w", encoding="utf-8") as f:
        json.dump(generation_data, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(retrieval_data)} synthetic query/chunk pairs across {len(CATEGORIES)} categories")
    print("  -> data/retrieval.json")
    print("  -> data/generation.json")


def generate_routing_data(per_tool=6):
    routing_data = []
    for tool, description in TOOL_DESCRIPTIONS.items():
        prompt = f"""اكتب {per_tool} أسئلة عربية متنوعة وواقعية، كل سؤال منها يطابق هذا الوصف بدقة:

{description}

اكتب كل سؤال في سطر منفصل، دون ترقيم أو أي نص إضافي."""
        result = llm.invoke(prompt)
        queries = [q.strip("- ").strip() for q in result.content.strip().split("\n") if q.strip()]
        for q in queries[:per_tool]:
            routing_data.append({"query": q, "expected_tool": tool})

    with open(os.path.join(DATA_DIR, "routing.json"), "w", encoding="utf-8") as f:
        json.dump(routing_data, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(routing_data)} synthetic routing queries")
    print("  -> data/routing.json")


if __name__ == "__main__":
    generate_routing_data(per_tool=6)
    generate_retrieval_and_generation_data(per_category=3)
