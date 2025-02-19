from dotenv import load_dotenv
import os
from pinecone import Pinecone

# Ladda API-nycklar
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY saknas i .env!")

# Initiera Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "csrd-index"

# Kontrollera om indexet finns
if INDEX_NAME not in pc.list_indexes().names():
    raise ValueError(f"❌ Index {INDEX_NAME} saknas! Skapa det först.")

index = pc.Index(INDEX_NAME)

# Hämta metadata om indexet
stats = index.describe_index_stats()

# Skriv ut info
print(f"📊 Indexstatistik: {stats}")

# Kontrollera om några vektorer har lagrats
total_vectors = stats.get("total_vector_count", 0)
if total_vectors == 0:
    print("⚠️ Inga vektorer finns i indexet! Något gick fel vid uppladdningen.")
else:
    print(f"✅ Indexet innehåller {total_vectors} vektorer.")

# Kontrollera att dimensionerna är korrekta (ska vara 1536 för text-embedding-ada-002)
dimension = stats.get("dimension", "Okänd")
if dimension != 1536:
    print(f"⚠️ Indexet har fel dimension ({dimension})! Det borde vara 1536.")
else:
    print("✅ Indexets dimensioner är korrekta (1536).")
