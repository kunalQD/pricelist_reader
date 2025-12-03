import pdfplumber
import pandas as pd

def extract_price_list(file_path, output_csv="final_price_list.csv"):
    records = []

    with pdfplumber.open(file_path) as pdf:
        table_list = []
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    table_list.append(row)

        columns = table_list[0]
        df = pd.DataFrame(table_list[1:], columns=columns)
    
    target_cols = {
    "file_name": ["file name", "filename", "file"],
    "cut_rate": ["cut rate", "cut"],
    "gst": ["gst", "hsn code gst", "hsn code"]
    }

    # Step 3: Find best match
    selected = {}
    for key, patterns in target_cols.items():
        for col in df.columns:
            if any(p in col for p in patterns):
                selected[key] = col
                break

    # Step 4: Subset DataFrame with required columns
    result = df[[selected["file_name"], selected["cut_rate"], selected["gst"]]]


    return result


# Example usage
pdf_path = "SPECIAL PRISE LIST - 2025.pdf"
final_df = extract_price_list(pdf_path)

print(final_df.head())
