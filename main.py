import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from vectorizer import vectorize_and_store
import uvicorn
from reviewer_ai import generate_reviewer_suggestions


app = FastAPI()

class VectorizeRequest(BaseModel):
    id: str
    software: str
    body: Dict[str, Any]

@app.post("/vectorize")
async def vectorize(request: VectorizeRequest):
    try:
        print("Received vectorize request:", request)
        result = vectorize_and_store(request.id, request.software, request.body)
        return {"status": "success", "data": result}
    except Exception as e:
        print("Exception occurred:")
        traceback.print_exc()  # <--- Print full error stack trace
        raise HTTPException(status_code=500, detail=str(e))


class ReviewerRequest(BaseModel):
    id: str

@app.post("/reviewerai")
async def reviewer_ai(request: ReviewerRequest):
    suggestions = generate_reviewer_suggestions(request.id)
    return suggestions

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)
