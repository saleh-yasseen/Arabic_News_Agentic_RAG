from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import sys
import traceback
import json

NODE_LABELS = {
    "route": "تحديد الأداة المناسبة...",
    "execute_tool": "البحث في الأرشيف والمصادر...",
    "retry_search": "إعادة الاستعلام لتحسين النتائج...",
    "generate": "صياغة الإجابة...",
}

executor = ThreadPoolExecutor(max_workers=4)

sys.path.append(r"S:\my_projects\Arabic_News_Agentic_RAG")
from agent.graph import app as agent_app
api=FastAPI()

class QueryRequest(BaseModel):
    query: str

def stream_agent(query: str):
    initial_state = {
        "query": query,
        "tool_choice": None,
        "context": "",
        "response": None,
        "loop_count": 0,
        "sources": [],
        "comparison": None
    }
    final_state = {}
    try:
        for step in agent_app.stream(initial_state, stream_mode="updates"):
            node_name = list(step.keys())[0]
            node_output = step[node_name]
            label = NODE_LABELS.get(node_name, node_name)

            yield f"data: {json.dumps({'type': 'status', 'node': node_name, 'label': label}, ensure_ascii=False)}\n\n"
            final_state.update(node_output)

        result_payload = {
            'type': 'result',
            'response': final_state.get('response', ''),
            'tool_used': final_state.get('tool_choice'),
            'loop_count': final_state.get('loop_count', 0),
            'sources': final_state.get('sources', []),
            'comparison': final_state.get('comparison')
        }
        yield f"data: {json.dumps(result_payload, ensure_ascii=False)}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"


@api.post("/query/stream")
def query_stream(request: QueryRequest):
    return StreamingResponse(
    stream_agent(request.query),
    media_type="text/event-stream",
    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}
)

@api.post("/query")
def query(request: QueryRequest):

    try:
        initial_state = {
            "query": request.query,
            "tool_choice": None,
            "context": "",
            "response": None,
            "loop_count": 0,
            "sources": [],
            "comparison": None
        }

        future = executor.submit(agent_app.invoke, initial_state)
        result = future.result(timeout=30)

        return{
            "query": result.get("query", request.query),
            "tool_choice": result.get("tool_choice", "unknown"),
            "context": result.get("context", ""),
            "response": result.get("response", ""),
            "loop_count": result.get("loop_count", 0),
            "sources": result.get("sources", []),
            "comparison": result.get("comparison")
        }
    
    except TimeoutError:
        return {"error": "timeout", "response": "استغرق الطلب وقتًا طويلاً، حاول مرة أخرى."}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e),"response": "حدث خطأ أثناء معالجة طلبك، حاول مرة أخرى."}

@api.get("/health")
def health():
    return {"status": "ok"}