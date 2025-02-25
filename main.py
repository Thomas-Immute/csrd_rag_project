import os
from fastapi import FastAPI, HTTPException, Request
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

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
    allow_origins=["*"],  
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

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Render sätter PORT, fallback till 8000
    uvicorn.run(app, host="0.0.0.0", port=port)

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
@app.post("/add-vector/")
async def add_vector(request: Request):
    """ Tar emot JSON-data från Squarespace och lägger till en vektor i Pinecone """
    try:
        data = await request.json()
        print("Mottagen data från Squarespace:", data)  # Logga inkommande data

        # Kontrollera att nödvändiga fält finns i datan
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="Felaktigt format: Data måste vara ett JSON-objekt.")

        if "id" not in data or "vector" not in data:
            raise HTTPException(status_code=400, detail="Felaktigt format: 'id' och 'vector' krävs.")

        vector_id = data["id"]
        vector_values = data["vector"]

        # Kontrollera vektorns dimension
        expected_dimension = 1536  # Anpassa efter ditt index
        if not isinstance(vector_values, list) or len(vector_values) != expected_dimension:
            raise HTTPException(status_code=400, detail=f"Felaktig vektordimension: {len(vector_values)} istället för {expected_dimension}.")

        # Lägg till vektorn i Pinecone
        index.upsert([(vector_id, vector_values)])

        return {"status": "success", "message": f"Vektor {vector_id} har lagts till."}

    except HTTPException as e:
        print(f"HTTP-fel: {e.detail}")
        raise e
    except Exception as e:
        print(f"Fel vid tillägg av vektor: {e}")
        raise HTTPException(status_code=500, detail=f"Internt serverfel: {str(e)}")

