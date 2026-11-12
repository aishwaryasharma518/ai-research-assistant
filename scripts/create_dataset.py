import fitz
import os
import pandas as pd

# Path to your dataset folder
DATASET_PATH = "data/dataset"

# Categories (folder names)
CATEGORIES = [
    "research_papers",
    "resumes",
    "legal",
    "medical",
    "business_report",
    "books"
]

def extract_text(pdf_path):
    text = ""

    try:
        doc = fitz.open(pdf_path)

        for page in doc:
            text += page.get_text()

        doc.close()

    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")

    return text


data = []

for category in CATEGORIES:

    folder = os.path.join(DATASET_PATH, category)

    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        continue

    print(f"\nReading {category}...")

    for file in os.listdir(folder):

        if file.lower().endswith(".pdf"):

            pdf_path = os.path.join(folder, file)

            print("Processing:", file)

            text = extract_text(pdf_path)

            data.append({
                "text": text,
                "label": category
            })

# Create DataFrame
df = pd.DataFrame(data)

# Save CSV
df.to_csv("data/dataset.csv", index=False)

print("\nDataset Created Successfully!")
print(df.head())
print("\nTotal Documents:", len(df))