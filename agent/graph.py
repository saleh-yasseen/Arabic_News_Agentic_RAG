from typing import TypedDict, Optional
from langchain_groq import ChatGroq
import groq
import json
from agent.tools import search_news, summarize_topic, compare_timeline, answer_direct
from dotenv import load_dotenv
import os
import time
import re
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

class AgentState(TypedDict):
    query: str
    tool_choice:Optional[str]
    context: str
    response:Optional[str]
    loop_count:int
    sources: list
    comparison: Optional[dict]
    model: str
    tool_override: Optional[str]
    retrieval_mode: str
    temperature: float
    top_k: int
    category_filter: Optional[str]
    use_reranker: bool
    min_score: Optional[float]

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

ROUTING_PROMPT = """
أنت وكيل توجيه لنظام أخبار عربي. بناءً على سؤال المستخدم، اختر أداة واحدة بالضبط:

- search_news: سؤال محدد عن حدث إخباري أو خبر بعينه
- summarize_topic: طلب نظرة عامة أو ملخص واسع عن موضوع إخباري جارٍ
- compare_timeline: سؤال عن كيف تطور موضوع عبر الزمن، أو مقارنة بين فترتين أو حدثين
- answer_direct: سؤال معرفة عامة، غير مرتبط بالأخبار الجارية (تعريف، حقيقة تاريخية، مفهوم عام)

أمثلة:

Query: ماذا قال الرئيس في خطابه أمس؟
Tool: search_news

Query: من فاز بالانتخابات البرلمانية في لبنان هذا الأسبوع؟
Tool: search_news

Query: من هو مخترع الهاتف؟
Tool: answer_direct

Query: ما هو الذكاء الاصطناعي؟
Tool: answer_direct

Query: أعطني نظرة عامة عن الوضع في سوريا
Tool: summarize_topic

Query: لخص لي آخر الأخبار الاقتصادية هذا الشهر
Tool: summarize_topic

Query: ما هو التضخم الاقتصادي؟
Tool: answer_direct

Query: كيف تطورت العلاقات الأمريكية الإيرانية عبر السنوات؟
Tool: compare_timeline

Query: قارن بين الوضع الاقتصادي قبل وبعد الجائحة
Tool: compare_timeline

Query: ما هي آخر التطورات في الأزمة اليمنية؟
Tool: search_news

الآن صنّف هذا السؤال:

Query: {query}

أجب باسم الأداة فقط، دون أي نص إضافي.
"""

GENERATION_PROMPT = """
أنت محرر أخبار عربي محترف. اكتب تقريرًا إخباريًا شاملاً باللغة العربية بناءً على السياق أدناه.

التنسيق المطلوب:
- عنوان رئيسي للخبر
- مقدمة تلخص أبرز ما جاء في الأخبار
- فقرة تتناول التفاصيل والسياق
- فقرة تتناول التداعيات أو الموقف الراهن
- أسلوب صحفي احترافي ومفصل

السياق: {context}

السؤال: {query}

التقرير الإخباري:
"""
SUMMARY_PROMPT = """
أنت محرر أخبار عربي. لخص الموضوع التالي بناءً على عدة مصادر مختلفة، مع إبراز النقاط الرئيسية المشتركة بين المصادر ونقاط الاختلاف إن وجدت.

السياق (من عدة مصادر):
{context}

السؤال: {query}

قدم ملخصاً منظماً على شكل نقاط رئيسية، وليس مقالاً إخبارياً مفرداً.
"""

def call_llm_with_retry(llm_instance, prompt, max_retries=5, initial_delay=10):
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return llm_instance.invoke(prompt)
        except Exception as e:
            msg = str(e)
            if "tokens per day" in msg or "TPD" in msg:
                match = re.search(r"try again in (\d+)m([\d.]+)s", msg)
                if match:
                    wait = int(match.group(1)) * 60 + float(match.group(2))
                    print(f"  daily token cap hit, waiting {wait:.0f}s (Groq's own estimate)")
                    time.sleep(wait + 2)
                    continue
                print("  daily token cap hit, no wait estimate parsed — stopping retries")
                raise
            if "429" in msg and attempt < max_retries - 1:
                print(f"  rate-limited, waiting {delay}s (attempt {attempt+2}/{max_retries})")
                time.sleep(delay)
                delay *= 1.5
                continue
            raise

def route_query(state: AgentState) -> dict:
    if state.get("tool_override"):
        return {"tool_choice": state["tool_override"]}
    llm = ChatGroq(model=state.get("model", "llama-3.1-8b-instant"), temperature=0)
    prompt = ROUTING_PROMPT.format(query=state["query"])
    result = call_llm_with_retry(llm, prompt, max_retries=10, initial_delay=10)
    tool_name = result.content.strip()
    valid_tools= ["search_news", "summarize_topic", "compare_timeline", "answer_direct"]
    if tool_name not in valid_tools:
        tool_name = "search_news"
    print("routing done")
    return {"tool_choice": tool_name}


TOOL_MAP = {
    "search_news":search_news,
    "summarize_topic":summarize_topic,
    "compare_timeline":compare_timeline,
    "answer_direct":answer_direct

}

def execute_tool(state: AgentState) -> dict:
    tool_fn = TOOL_MAP[state["tool_choice"]]
    tool_name = state["tool_choice"]
    if tool_name == "search_news":
        result = tool_fn(
            state["query"], mode=state.get("retrieval_mode", "hybrid"),
            top_k=state.get("top_k", 5), category_filter=state.get("category_filter"),
            use_reranker=state.get("use_reranker", True), min_score=state.get("min_score")
        )
    elif tool_name == "summarize_topic":
        result = tool_fn(state["query"], top_k=8, use_reranker=state.get("use_reranker", True))
    elif tool_name == "compare_timeline":
        result = tool_fn(state["query"], category=state.get("category_filter"), top_k=10)
    else:
        result = tool_fn(state["query"])

    context = result.get("context") or " ".join(r["text"] for r in result.get("results", []))

    return {
        "context": context,
        "sources": result.get("results", []),
        "comparison": result.get("comparison"),
    }


def generate_response(state: AgentState) -> dict :
    llm = ChatGroq(model=state.get("model", "llama-3.3-70b-versatile"), temperature=state.get("temperature", 0.0))
    tool = state.get("tool_choice")
    template = SUMMARY_PROMPT if tool == "summarize_topic" else GENERATION_PROMPT
    prompt = template.format(context=state["context"], query=state["query"])
    result = call_llm_with_retry(llm, prompt, max_retries=10, initial_delay=10)
    print("generation_done")
    return {"response":result.content}

def check_context_quality(state: AgentState) ->str :
    if state["loop_count"] >= 2:
        return "generate"
    if len(state.get("sources", [])) < 3:
        return "retry"
    print("loop is working")
    return "generate"

def retry_search(state: AgentState) -> dict :
    return {
        "loop_count": state["loop_count"] + 1,
        "tool_choice":"search_news"
    }


from langgraph.graph import StateGraph ,END

graph = StateGraph(AgentState)

graph.add_node("route",route_query)
graph.add_node("execute_tool", execute_tool)
graph.add_node("retry_search", retry_search)
graph.add_node("generate", generate_response)

graph.set_entry_point("route")
graph.add_edge("route", "execute_tool")

graph.add_conditional_edges(
    "execute_tool",
    check_context_quality,
    {"retry": "retry_search", "generate": "generate"}
)

graph.add_edge("retry_search", "execute_tool")
graph.add_edge("generate", END)

app = graph.compile()

if __name__ =="__main__":
    initial_state ={
        "query": "ما هي آخر التطورات في الوضع السوري؟",
        "tool_choice": None,
        "context" : "",
        "response": None,
        "loop_count" : 0
    }

    result = app.invoke(initial_state)
    print(result)

# graph_image = app.get_graph().draw_mermaid_png()
# with open("agent_graph.png", "wb") as f:
#     f.write(graph_image)
