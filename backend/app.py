from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag import RAGEngine  # Custom RAG class (see below)
import os

app = FastAPI()

# Load RAG engine
rag_engine = RAGEngine(data_dir="backend/data")

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        answer = rag_engine.query(request.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
