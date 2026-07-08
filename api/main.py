from fastapi import FastAPI
from pydantic import BaseModel
import sys
sys.path.append(r"S:\my_projects\Arabic_News_Agentic_RAG")
from agent.graph import app as agent_app
api=FastAPI()

class QueryRequest(BaseModel):
    query: str

@api.post("/query")
def query(request: QueryRequest):
    initial_state = {
        "query": request.query,
        "tool_choice": None,
        "context": None,
        "response": None,
        "loop_count": 0
    }
    
    result = agent_app.invoke(initial_state)
    return{
        "query": result["query"],
        "tool_choice": result["tool_choice"],
        "context": result["context"],
        "response": result["response"],
        "loop_count": result["loop_count"]
    }

@api.get("/health")
def health():
    return {"status": "ok"}