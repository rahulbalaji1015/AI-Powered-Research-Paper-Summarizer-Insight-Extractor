import pandas as pd
import json
import os

# ---------------------------------------------------
# Function: excel_to_json
# ---------------------------------------------------
def excel_to_json(excel_path, json_path="output.json"):
    if not os.path.exists(excel_path):
        print("❌ File not found:", excel_path)
        return

    try:
        # Read Excel file into a pandas DataFrame
        df = pd.read_excel(excel_path)

        # ✅ Remove columns like "Unnamed: 0" (usually index columns)
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

        # ✅ Remove completely empty rows
        df = df.dropna(how="all")

    except PermissionError:
        # Handle case where Excel file is open or access is denied
        print("❌ Permission denied. Please close the Excel file and try again.")
        return

    # Convert DataFrame into list of dictionaries (JSON-like structure)
    data = df.to_dict(orient="records")

    # Write the JSON data to output file
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print("✅ Clean JSON saved to", json_path)


# ---------------------------------------------------
# Function: main
# ---------------------------------------------------
def main():
    excel_path = input("Enter Excel file path: ").strip().replace('"','')
    json_path = input("Enter output JSON file name: ").strip()

    excel_to_json(excel_path, json_path)
    
# ---------------------------------------------------
# main() function
# ---------------------------------------------------
if __name__ == "__main__":
    main()
