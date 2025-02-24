import os
from fastapi import FastAPI, HTTPException, Request
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Ladda miljövariabler från .env
load_dotenv(".env")

# Hämta API-nycklar och miljövariabler från Render
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", 1536))  # Standardvärde
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Kontrollera att API-nycklar finns
if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
    raise RuntimeError("Pinecone API-nyckel eller indexnamn saknas. Kontrollera dina miljövariabler.")

# Skapa en instans av FastAPI
app = FastAPI()

# Lägg till CORS-stöd
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://finch-penguin-z9kz.squarespace.com"],  # ✅ Ta bort extra "/"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initiera Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Funktion för att hämta eller skapa indexet
def get_or_create_index():
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=VECTOR_DIMENSION,  
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",  
                region="us-east-1"
            ),
        )
    return pc.Index(PINECONE_INDEX_NAME)

# Hämta indexet en gång vid start
index = get_or_create_index()

# Endpoint för att verifiera att API:t fungerar
@app.get("/")
def read_root():
    return {"message": "API fungerar!"}

# Definiera en modell för vektor-data
class Vector(BaseModel):
    id: str
    values: list[float]

# Endpoint för att lägga till en vektor i Pinecone
@app.post("/add-vector/")
def add_vector(vector: Vector):
    try:
        index.upsert(vectors=[{"id": vector.id, "values": vector.values}])
        return {"message": f"Vektor {vector.id} har lagts till."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fel vid tillägg av vektor: {str(e)}")

# Endpoint för att söka efter en vektor i Pinecone
@app.post("/search/")
def search_vector(values: list[float], top_k: int = 5):
    try:
        result = index.query(vector=values, top_k=top_k, include_metadata=True)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fel vid sökning: {str(e)}")

@app.post("/add-vector/")
async def add_vector(request: Request):
    # Hämta data från Squarespace-förfrågan
    data = await request.json()

    # Logga inkommande data i terminalen
    print("Mottagen data från Squarespace:", data)

    # Returnera ett test-svar för att se om API:et fungerar
    return {"status": "ok"}
