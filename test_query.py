import os
import pinecone
import openai

# Initiera Pinecone
pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("csrd-index")

# Testfråga
query_text = "Vad innebär CSRD för små företag?"

# Skapa embedding för frågan
openai.api_key = os.getenv("OPENAI_API_KEY")
response = openai.embeddings.create(
    input=query_text,
    model="text-embedding-ada-002"
)
query_embedding = response.data[0].embedding

# Söka i Pinecone
search_results = index.query(
    vector=query_embedding,
    top_k=3,  # Hämta relevanta träffar
    include_metadata=True
)

# Samla ihop och organisera träffar
organized_results = []
for match in search_results["matches"]:
    chunk_id = match["metadata"].get("chunk_id", None)
    doc_name = match["metadata"].get("document_name", "Okänt dokument")
    section = match["metadata"].get("section", "Okänt avsnitt")
    page = match["metadata"].get("page", "N/A")
    timestamp = match["metadata"].get("timestamp", "N/A")
    text = match["metadata"].get("text", "")

    # Hämta angränsande chunks
    adjacent_chunks = []
    if chunk_id is not None:
        # Hämta föregående chunk
        prev_chunk = index.query(
            vector=query_embedding,
            filter={"chunk_id": chunk_id - 1, "document_name": doc_name},
            top_k=1,
            include_metadata=True
        )
        # Hämta nästa chunk
        next_chunk = index.query(
            vector=query_embedding,
            filter={"chunk_id": chunk_id + 1, "document_name": doc_name},
            top_k=1,
            include_metadata=True
        )

        if prev_chunk["matches"]:
            adjacent_chunks.append(prev_chunk["matches"][0]["metadata"]["text"])
        if next_chunk["matches"]:
            adjacent_chunks.append(next_chunk["matches"][0]["metadata"]["text"])

    # Kombinera angränsande chunks
    full_text = "\n".join([t for t in [*adjacent_chunks, text] if t])

    # Organisera resultat
    organized_results.append(
        f"""
📄 **Dokument:** {doc_name}
📌 **Avsnitt:** {section}
📜 **Sida:** {page}
⏳ **Tidsstämpel:** {timestamp}

✍️ **Textutdrag:**  
{full_text}
---
""")

# Slå ihop resultaten
final_text = "\n\n".join(organized_results)

# Använd GPT för att skapa ett sammanhängande svar
summary_response = openai.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "Sammanfatta och förklara svaret tydligt."},
        {"role": "user", "content": f"Fråga: {query_text}\n\nRelevanta utdrag:\n{final_text}"}
    ]
)

# Visa förbättrat svar
print("\n🔍 **Sammanfattat svar:**")
print(summary_response.choices[0].message.content)
