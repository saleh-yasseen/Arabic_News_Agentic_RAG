from fastapi import FastAPI
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import sys
import traceback

executor = ThreadPoolExecutor(max_workers=4)

sys.path.append(r"S:\my_projects\Arabic_News_Agentic_RAG")
from agent.graph import app as agent_app
api=FastAPI()

class QueryRequest(BaseModel):
    query: str

@api.post("/query")
def query(request: QueryRequest):

    try:
        initial_state = {
            "query": request.query,
            "tool_choice": None,
            "context": "",
            "response": None,
            "loop_count": 0,
            "sources": []
        }

        future = executor.submit(agent_app.invoke, initial_state)
        result = future.result(timeout=30)

        return{
            "query": result.get("query", request.query),
            "tool_choice": result.get("tool_choice", "unknown"),
            "context": result.get("context", ""),
            "response": result.get("response", ""),
            "loop_count": result.get("loop_count", 0),
            "sources": result.get("sources", [])
        }
    
    except TimeoutError:
        return {"error": "timeout", "response": "استغرق الطلب وقتًا طويلاً، حاول مرة أخرى."}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e),"response": "حدث خطأ أثناء معالجة طلبك، حاول مرة أخرى."}

@api.get("/health")
def health():
    return {"status": "ok"}