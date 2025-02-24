import os
from fastapi import FastAPI, HTTPException
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Skapa en instans av FastAPI
app = FastAPI()

load_dotenv()

# Hämta API-nycklar och miljövariabler från Render
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", 1536))  # Använd standardvärde om ingen är satt

# Kontrollera att API-nycklar finns
if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
    raise RuntimeError("Pinecone API-nyckel eller indexnamn saknas. Kontrollera dina miljövariabler.")

# Initiera Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Kontrollera om indexet finns, annars skapa det
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=VECTOR_DIMENSION,  # Viktigt att dimension anges här
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",  # Justera vid behov
            region="us-east-1"  # Justera vid behov
        ),
    )

# Funktion för att hämta indexet
def get_index():
    try:
        return pc.Index(PINECONE_INDEX_NAME)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fel vid anslutning till Pinecone-index: {str(e)}")

# Endpoint för att verifiera att API:t fungerar
@app.get("/")
def read_root():
    return {"message": "API fungerar!"}

# Endpoint för att lägga till en vektor i Pinecone
@app.post("/add-vector/")
def add_vector(id: str, values: list[float]):
    try:
        index = get_index()
        index.upsert(vectors=[{"id": id, "values": values}])
        return {"message": f"Vektor {id} har lagts till."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fel vid tillägg av vektor: {str(e)}")

# Endpoint för att söka efter en vektor i Pinecone
@app.post("/search/")
def search_vector(values: list[float], top_k: int = 5):
    try:
        index = get_index()
        result = index.query(vector=values, top_k=top_k, include_metadata=True)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fel vid sökning: {str(e)}")
