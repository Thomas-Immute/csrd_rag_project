import os
from fastapi import FastAPI, HTTPException
from pinecone import Pinecone
from pydantic import BaseModel
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Hämta miljövariabler från Render
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initiera API:er
client = OpenAI(api_key=OPENAI_API_KEY) 
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Skapa FastAPI-app
app = FastAPI()

# Lägg till CORS-stöd och tillåt endast din domän
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.csrd-guiden.net"],  # Ändra till din domän
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rot-endpoint
@app.get("/")
async def read_root():
    return {"message": "API fungerar!"}

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
        response = client.embeddings.create(
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
    print(f"Mottagen data: {data}")
    try:
        response = client.embeddings.create(
            input=data.message,
            model="text-embedding-ada-002"
        )
        query_vector = response.data[0].embedding
        print(f"Embedding skapad: {query_vector[:5]}...")

        search_results = index.query(vector=query_vector, top_k=3, include_metadata=True)
        print(f"Pinecone-resultat: {search_results}")

        if search_results["matches"] and search_results["matches"][0]["score"] > 0.7:
            best_match = search_results["matches"][0]
            best_match_chunk_id = int(best_match["metadata"]["chunk_id"])
            best_match_document_id = best_match["metadata"]["document_id"]

            context = best_match["metadata"]["text"] + "\n"

            # Hämta angränsande vektorer
            for match in search_results["matches"]:
                if match["metadata"]["document_id"] == best_match_document_id:
                    match_chunk_id = int(match["metadata"]["chunk_id"])
                    if match_chunk_id == best_match_chunk_id - 1:
                        context = match["metadata"]["text"] + "\n" + context
                    elif match_chunk_id == best_match_chunk_id + 1:
                        context = context + match["metadata"]["text"] + "\n"

            # GPT-4 tolkar databasens svar
            gpt_response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": f"Du är en expert på CSRD och ESRS. Svara på användarens fråga baserat på följande kontext: {context}"},
                    {"role": "user", "content": data.message}
                ]
            )
            print(f"GPT-4-svar: {gpt_response}")
            return {"response": gpt_response.choices[0].message.content, "source": "database"}

        # Ingen matchning hittades, använd GPT-4 direkt
        gpt_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Du är en expert på CSRD och ESRS. Svara på användarens fråga."},
                {"role": "user", "content": data.message}
            ]
        )
        print(f"GPT-4-svar: {gpt_response}")
        return {"response": gpt_response.choices[0].message.content, "source": "gpt-4"}

    except Exception as e:
        print(f"Fel vid sökning: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Fel vid sökning: {str(e)}")