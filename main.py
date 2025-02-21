import os
from fastapi import FastAPI
from pinecone import Pinecone

# Skapa en instans av FastAPI
app = FastAPI()

# Hämta API-nycklar och miljövariabler från Render
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initiera Pinecone med den nya metoden
pc = Pinecone(api_key=PINECONE_API_KEY)

# Kontrollera om indexet finns, annars skapa det
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,  # Anpassa detta beroende på din embedding-modell
        metric="cosine"
    )

# Endpoint för att verifiera att API:t fungerar
@app.get("/")
def read_root():
    return {"message": "API fungerar!"}

# Endpoint för att lägga till en vektor i Pinecone
@app.post("/add-vector/")
def add_vector(id: str, values: list):
    index = pc.Index(PINECONE_INDEX_NAME)
    index.upsert(vectors=[{"id": id, "values": values}])
    return {"message": f"Vektor {id} har lagts till."}

# Endpoint för att söka efter en vektor i Pinecone
@app.post("/search/")
def search_vector(values: list, top_k: int = 5):
    index = pc.Index(PINECONE_INDEX_NAME)
    result = index.query(vector=values, top_k=top_k, include_metadata=True)
    return result
