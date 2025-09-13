import streamlit as st
import pandas as pd
import glob
import os

st.set_page_config(page_title="Pricelist Search", layout="wide")

st.title("🔎 Pricelist Search Tool")

# Find all CSV files in the current directory
csv_files = glob.glob("*.csv")

if not csv_files:
    st.error("⚠️ No CSV files found in this folder.")
else:
    # Map file paths to clean brand names
    brand_map = {
        file: os.path.basename(file).replace("Pricelist - New - ", "").replace(".csv", "")
        for file in csv_files
    }

    # Dropdown with brand names
    selected_brand = st.selectbox("Select a Brand", list(brand_map.values()))

    if selected_brand:
        # Get the actual file path
        selected_file = [file for file, brand in brand_map.items() if brand == selected_brand][0]

        # ✅ Use second row (index 1) as header
        df = pd.read_csv(selected_file)

        # Search input
        query = st.text_input("Search for an item")

        if query:
            filtered_df = df[df.apply(
                lambda row: row.astype(str).str.contains(query, case=False, na=False).any(),
                axis=1
            )]
            st.write(f"Showing results for **{query}** in **{selected_brand}**:")
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)  # 🚀 Hide index
        else:
            st.write(f"Showing full pricelist for **{selected_brand}**:")
            st.dataframe(df, use_container_width=True, hide_index=True)  # 🚀 Hide index
