from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from main import retrieve_answer_from_docs

app = FastAPI()

class QueryRequest(BaseModel):
    question: str



@app.post("/ask/")
async def ask_question(query: QueryRequest):
    # Validate the input
    if not query.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Call the main logic
    answer = retrieve_answer_from_docs(query.question)
    
    return {"question": query.question, "answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)