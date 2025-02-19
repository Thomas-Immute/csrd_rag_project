import os
import json
import pinecone
import openai
from pinecone import Pinecone
from dotenv import load_dotenv

# Ladda miljövariabler
load_dotenv()

# Hämta API-nycklar
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not PINECONE_API_KEY or not PINECONE_ENV:
    raise ValueError("Pinecone API-nyckel eller miljö saknas i .env-filen!")

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API-nyckel saknas i .env-filen!")

# Initiera Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "csrd-index"

available_indexes = pc.list_indexes().names()
print(f"📌 Tillgängliga index i Pinecone: {available_indexes}")

index = pc.Index(INDEX_NAME)  # Hämta indexet
print(f"✅ Indexinstans skapad: {index}")

# Kontrollera att indexet finns
if INDEX_NAME not in pc.list_indexes().names():
    print(f"⚠️ Index '{INDEX_NAME}' saknas. Skapa det först!")
    exit()

index = pc.Index(INDEX_NAME)

# Initiera OpenAI
openai.api_key = OPENAI_API_KEY

# Funktion för att generera embeddings
def generate_embedding(text):
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

# Mapp med uppdelade textfiler i JSON-format
input_folder = "chunks"

for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)

    # Säkerställ att det är en JSON-fil
    if not filename.endswith(".json"):
        continue

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            doc = json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ Fel vid läsning av {filename}: {e}")
            continue

    # Kontrollera att JSON-filen har rätt format
    if not isinstance(doc, dict):
        print(f"❌ Fel format i {filename}: Förväntade en dictionary men fick {type(doc)}")
        continue
    if "id" not in doc or "text" not in doc or "source" not in doc:
        print(f"❌ Filen saknar nödvändiga nycklar: {filename}")
        continue

    text = doc["text"]
    doc_id = doc["id"]
    source = doc["source"]

    # Skapa embedding från OpenAI
    try:
        embedding = generate_embedding(text)
    except Exception as e:
        print(f"❌ Fel vid generering av embedding för {filename}: {e}")
        continue

    # Debug-utskrift
    print(f"🔹 Uppladdar dokument: {doc_id}")
   
    # Skicka upp till Pinecone
    try:
        index.upsert([(doc_id, embedding, {"text": text, "source": source})])
        print(f"✅ Uppladdad: {filename}")
    except Exception as e:
        print(f"❌ Fel vid uppladdning av {filename}: {e}")

print("🚀 Alla filer har bearbetats!")
