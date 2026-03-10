import PyPDF2
import json
import re
import os

JSON_FILE = "core_data.json"

# ---------------------------------------------------
# Function: extract_text_from_pdf
# ---------------------------------------------------
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

# ---------------------------------------------------
# Function: extract_sections
# ---------------------------------------------------
def extract_sections(text):
    section_titles = [
        "Introduction",
        "Related Work",
        "Literature Review",
        "Methodology",
        "Methods",
        "Proposed System",
        "Implementation",
        "Results",
        "Discussion",
        "Conclusion",
        "Future Work",
    ]

    sections = {}
    pattern = r"(?=(" + "|".join(section_titles) + r"))"
    parts = re.split(pattern, text, flags=re.IGNORECASE)

    current_section = "Paper Details"

    for part in parts:
        part = part.strip()
        if part.capitalize() in section_titles:
            current_section = part.capitalize()
            sections[current_section] = ""
        else:
            if current_section not in sections:
                sections[current_section] = ""
            sections[current_section] += part + " "

    return sections

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
def save_paper_data(paper_id, sections):
    data = load_existing_data()

    paper_entry = {
        "Paper_ID": paper_id,
        "sections": sections
    }

    data.append(paper_entry)

    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"✅ Paper {paper_id} saved successfully!")

# ---------------------------------------------------
# Function: main
# ---------------------------------------------------
def main():
    paper_id = input("Enter Paper ID (P01, P02, etc.): ")
    pdf_path = input("Enter path of research paper PDF: ")

    print("📄 Extracting text...")
    text = extract_text_from_pdf(pdf_path)

    print("📑 Extracting sections...")
    sections = extract_sections(text)

    save_paper_data(paper_id, sections)


if __name__ == "__main__":
    main()
