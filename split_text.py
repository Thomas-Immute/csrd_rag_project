import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

input_folder = "extracted_text"
output_folder = "chunks"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(input_folder, filename)

        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
        chunks = text_splitter.split_text(text)

        for i, chunk in enumerate(chunks):
            chunk_data = {
                "id": f"{filename}_chunk{i}",
                "text": chunk,
                "source": filename,
                "chunk_id": i,  # Lägg till chunk_id
                "document_id": filename  # Lägg till document_id
            }

            json_path = os.path.join(output_folder, f"{filename}_chunk{i}.json")
            with open(json_path, "w", encoding="utf-8") as json_file:
                json.dump(chunk_data, json_file, ensure_ascii=False, indent=4)

print("Texten har delats upp och sparats som JSON!")