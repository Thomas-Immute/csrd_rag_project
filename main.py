import os
from fastapi import FastAPI, HTTPException
from pinecone import Pinecone
from dotenv import load_dotenv
from pydantic import BaseModel
import openai

# Ladda miljövariabler
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initiera API:er
openai.api_key = OPENAI_API_KEY
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Skapa FastAPI-app
app = FastAPI()

# Modell för att ta emot data
class MessageInput(BaseModel):
    id: str
    message: str

# Modell för sökning
class SearchInput(BaseModel):
    message: str

# Endpoint för att lägga till vektor i Pinecone
@app.post("/add-vector/")
async def add_vector(data: MessageInput):
    try:
        # Skapa en embedding från meddelandet
        response = openai.Embedding.create(
            input=data.message,
            model="text-embedding-ada-002"
        )
        vector = response["data"][0]["embedding"]

        # Lagra vektorn i Pinecone
        index.upsert([{"id": data.id, "values": vector, "metadata": {"text": data.message}}])

        return {"status": "success", "message": f"Vektor för '{data.message}' har lagts till."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fel: {str(e)}")

# Endpoint för att söka i Pinecone
@app.post("/search/")
async def search_vector(data: SearchInput):
    try:
        # Skapa en embedding från sökfrågan
        response = openai.Embedding.create(
            input=data.message,
            model="text-embedding-ada-002"
        )
        query_vector = response["data"][0]["embedding"]

        # Sök i Pinecone
        search_results = index.query(vector=query_vector, top_k=1, include_metadata=True)

        # Om vi hittar en match, returnera den
        if search_results["matches"] and search_results["matches"][0]["score"] > 0.8:
            best_match = search_results["matches"][0]
            return {"response": best_match["metadata"]["text"]}

        # Om ingen bra match hittas, använd GPT-4
        gpt_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "Du är en expert på CSRD och ESRS."},
                      {"role": "user", "content": data.message}]
        )

        return {"response": gpt_response["choices"][0]["message"]["content"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fel vid sökning: {str(e)}")
