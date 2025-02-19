import os
import pdfplumber

input_folder = "docs"
output_folder = "extracted_text"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".pdf"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace(".pdf", ".txt"))

        with pdfplumber.open(input_path) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"✅ Extraherade text från: {filename}")
