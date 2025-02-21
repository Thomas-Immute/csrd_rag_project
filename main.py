import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pinecone 

app = FastAPI()

# Hämta API-nycklar från miljövariabler
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
INDEX_NAME = os.environ.get("INDEX_NAME")

# Initiera Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pinecone.Index(INDEX_NAME)

class ChatRequest(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"message": "API fungerar!"}


@app.post("/chat")
def chat(request: ChatRequest):
    question = request.question  

    # Skicka frågan till Pinecone och hämta svar
    response = index.query(queries=[question], top_k=1, include_metadata=True)
    
    if response and "matches" in response:
        answer = response["matches"][0]["metadata"].get("text", "Jag kunde inte hitta ett svar.")
    else:
        answer = "Jag kunde inte hitta ett svar."

    return {"answer": answer}