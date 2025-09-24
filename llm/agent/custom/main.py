from fastapi import FastAPI, HTTPException, Request
from typing import Dict, Any
from uuid import uuid4
from slowapi import Limiter
from slowapi.util import get_remote_address

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)

@app.post("/api/parallel-nodes", response_model=Dict[str, Any])
@limiter.limit("10/day")
async def run_parallel_nodes_graph(request: Request, input_data: Dict[str, Any]):
    """
    Endpoint to execute the parallelNodesGraph with custom input
    """
    try:
        thread_id = str(uuid4())
        context = {
            "a": input_data.get("a", 0),
            "b": input_data.get("b", 0),
            "name": input_data.get("name", "User")
        }
        from createAndCompileGraph import parallelNodesGraph
        result = await parallelNodesGraph(thread_id=thread_id, context=context)
        return {
            "status": "success",
            "thread_id": thread_id,
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

#############################################################
# curl -X POST "http://localhost:8000/api/parallel-nodes" \ 
#   -H "Content-Type: application/json" \                   
#   -d '{"a": 10, "b": 5, "name": "Test User"}'             
#############################################################
