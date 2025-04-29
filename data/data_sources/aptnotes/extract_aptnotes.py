import os
import json
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def save_text(output_path, text):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

def process_aptnotes_json(json_path, processed_folder):
    with open(json_path, "r", encoding="utf-8") as f:
        aptnotes = json.load(f)
    for entry in aptnotes:
        # Example: Save each entry as a separate JSON file
        filename = entry.get("Filename", "entry") + ".json"
        output_path = os.path.join(processed_folder, filename)
        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(entry, out_f, indent=2)
        print(f"Processed {filename}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, "raw", "data-master", "data-master", "APTnotes.json")
    processed_folder = os.path.join(base_dir, "processed")
    os.makedirs(processed_folder, exist_ok=True)
    process_aptnotes_json(json_path, processed_folder)
