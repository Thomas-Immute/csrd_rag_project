from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import pinecone
import os

# Initiera Flask
app = Flask(__name__)
CORS(app)  # Möjliggör förfrågningar från din Squarespace-webbplats

# Hämta API-nycklar från miljövariabler
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

# Initiera OpenAI
openai.api_key = OPENAI_API_KEY

# Initiera Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("csrd-index")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    query_text = data.get("question", "")

    if not query_text:
        return jsonify({"error": "Ingen fråga angiven"}), 400

    # Hämta relevanta vektorer från Pinecone
    query_embedding = openai.embeddings.create(input=query_text, model="text-embedding-ada-002").data[0].embedding
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

    # Extrahera text från träffarna och lägg till metadata
    context = "\n\n".join([match['metadata']['text'] for match in results["matches"]])
    
    # Skapa ett svar med OpenAI (GPT-4)
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Du är en expert på CSRD och hjälper användare att förstå rapportering."},
            {"role": "user", "content": f"Fråga: {query_text}\n\nRelevant information:\n{context}"}
        ]
    )

    answer = response["choices"][0]["message"]["content"]

    return jsonify({"answer": answer, "context": context})

# Starta servern
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
