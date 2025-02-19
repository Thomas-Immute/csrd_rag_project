from dotenv import load_dotenv
import os
from pinecone import Pinecone

# Ladda miljövariabler
load_dotenv()

# Initiera Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "csrd-index"

pc = Pinecone(api_key=PINECONE_API_KEY)

# Kontrollera att indexet finns
if INDEX_NAME not in pc.list_indexes().names():
    print(f"⚠️ Index {INDEX_NAME} finns inte.")
    exit()

index = pc.Index(INDEX_NAME)

# Skicka en test-query
query_result = index.query(
    vector=[0.0] * 1536,  # Placeholder för en slumpmässig query
    top_k=5,
    include_metadata=True
)

print("🔍 Pinecone-query resultat:", query_result)
query_result = index.query(
    vector=[0.0] * 1536,  # Placeholder för en slumpmässig query
    top_k=1,
    include_metadata=True
)
print("🔍 Pinecone-query resultat:", query_result)
