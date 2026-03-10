import PyPDF2
import json
import re
import os

JSON_FILE = "abstract_data.json"


# ---------------------------------------------------
# Function: extract_text_from_pdf
# ---------------------------------------------------
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text


# ---------------------------------------------------
# Function: extract_abstract
# ---------------------------------------------------
def extract_abstract(text):
    # Extract text between Abstract and Introduction/Keywords
    pattern = r"Abstract(.*?)(Keywords|Introduction|I\.|1\.)"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)

    if match:
        return match.group(1).strip()
    else:
        return "Abstract not found"


# ---------------------------------------------------
# Function: load_existing_data
# ---------------------------------------------------
def load_existing_data():
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


# ---------------------------------------------------
# Function: save_paper_data
# ---------------------------------------------------
def save_paper_data(paper_id, paper_name, abstract_text):
    data = load_existing_data()

    paper_entry = {
        "Paper_ID": paper_id,
        "abstract": abstract_text
    }

    data.append(paper_entry)

    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"✅ Paper {paper_id} ({paper_name}) saved successfully!")

def main():
    paper_id = input("Enter Paper ID (P01, P02, etc.): ").strip()
    pdf_path = input("Enter path of research paper PDF: ").strip().replace('"', '')

    # Extract file name as paper name
    paper_name = os.path.basename(pdf_path)

    print("📄 Extracting text...")
    text = extract_text_from_pdf(pdf_path)

    print("📑 Extracting Abstract section...")
    abstract_text = extract_abstract(text)

    save_paper_data(paper_id, paper_name, abstract_text)

if __name__ == "__main__":
    main()
