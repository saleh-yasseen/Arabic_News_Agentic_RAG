from typing import TypedDict, Optional
from langchain_groq import ChatGroq
import json
from agent.tools import search_news, summarize_topic, compare_timeline, answer_direct
from dotenv import load_dotenv
import os
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

class AgentState(TypedDict):
    query: str
    tool_choice:Optional[str]
    context: str
    response: str
    loop_count:int

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

ROUTING_PROMPT = """

you are a routing agent for an arabic news system. given a user query, choose exactly one tool:

- search_news: for specific factual questions about a news event or topic
- summarize_topic: for broad questions asking for an overview or summary of a topic
- compare_timeline: for questions asking about how something developed over time, or comparing events
- answer_direct: for general knowledge questions unrelated to news (definitions, concepts)

Query:{query}

respond with only one tool name, nothing else. 

"""

GENERATION_PROMPT = GENERATION_PROMPT = """
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

def route_query(state: AgentState) -> dict:
    prompt = ROUTING_PROMPT.format(query=state["query"])
    result =llm.invoke(prompt)
    tool_name =result.content.strip()
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
    tool_fn=TOOL_MAP[state["tool_choice"]]
    result = tool_fn(state["query"])

    if "context" in result:
        context = result["context"]
    elif "results" in result:
        context = " ".join([r["text"] for r in result["results"]])
    else:
        context = ""
    
    print("execute tool done")
    return{"context":context}


def generate_response(state: AgentState) -> dict :
    prompt = GENERATION_PROMPT.format(context=state["context"], query=state["query"])
    result = llm.invoke(prompt)
    print("generation_done")
    return {"response":result.content}

def check_context_quality(state: AgentState) ->str :
    if state["loop_count"] >= 2:
        return "generate"
    if len(state["context"]) < 100:
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