# streamlit_pricelist_from_pdf.py
import streamlit as st
import pandas as pd
import glob
import os
import io

st.set_page_config(page_title="Pricelist Search (PDF→CSV)", layout="wide")
st.title("🔎 Pricelist Search Tool — Read PDF & Generate New Pricelist")

st.markdown(
    """
    Upload or place PDF pricelists in the app folder. The app will try to extract tables from the PDF,
    let you choose the numeric column that contains the rate, then calculate a new price as:
    **new_price = round(rate * 1.05 * 2.5)** and let you download the generated pricelist CSV.
    """
)

# -----------------------
# Helper functions
# -----------------------
def try_import(module_name):
    try:
        return __import__(module_name)
    except Exception:
        return None

def extract_tables_with_tabula(path):
    """
    Uses tabula-py (requires Java). Returns concatenated dataframe or raises if not found/usable.
    """
    tabula = try_import("tabula")
    if not tabula:
        raise ImportError("tabula not installed")
    # read all tables from all pages; multiple_tables=True returns a list of DataFrames
    dfs = tabula.read_pdf(path, pages="all", multiple_tables=True)
    if not dfs:
        return None
    # Concatenate tables vertically (user can later choose header row)
    return pd.concat(dfs, ignore_index=True)

def extract_tables_with_pdfplumber(path):
    """
    A fallback using pdfplumber (pure python). Tries to extract tables page by page.
    """
    pdfplumber = try_import("pdfplumber")
    if not pdfplumber:
        raise ImportError("pdfplumber not installed")
    dflist = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            try:
                tables = page.extract_tables()
            except Exception:
                tables = None
            if not tables:
                continue
            for tab in tables:
                # tab is list-of-rows (each row is list of cells)
                if len(tab) < 1:
                    continue
                df = pd.DataFrame(tab)
                dflist.append(df)
    if not dflist:
        return None
    return pd.concat(dflist, ignore_index=True)

def sanitize_numeric_series(s: pd.Series) -> pd.Series:
    # remove commas, currency symbols, parentheses, non-digit except dot and minus
    return (
        s.astype(str)
         .str.replace(r"[^\d\.-]", "", regex=True)
         .replace("", pd.NA)
         .astype("float64", errors="ignore")
    )

def detect_numeric_columns(df: pd.DataFrame, max_candidates=10):
    numeric_cols = []
    for col in df.columns:
        # try to coerce a sample to numeric
        coerced = (
            df[col].astype(str)
            .str.replace(r"[^\d\.-]", "", regex=True)
            .replace("", pd.NA)
        )
        # count how many numeric-like entries exist
        num_count = coerced.dropna().str.match(r"^-?\d+(\.\d+)?$").sum()
        total_count = len(df[col].dropna())
        # heuristic: at least 30% numeric entries and at least 3 numeric values
        if total_count >= 3 and num_count / max(1, total_count) >= 0.3:
            numeric_cols.append(col)
        if len(numeric_cols) >= max_candidates:
            break
    return numeric_cols

# -----------------------
# Gather PDF files in current folder
# -----------------------
pdf_files = glob.glob("*.pdf")
st.write("### Files found in folder")
if not pdf_files:
    st.info("No PDF files found in this folder. You can upload one below.")
else:
    # Map file path to cleaned brand name (similar to your CSV logic)
    brand_map = {
        file: os.path.basename(file).replace("Pricelist - New - ", "").replace(".pdf", "")
        for file in pdf_files
    }
    selected_brand = st.selectbox("Select a Brand / PDF file", ["(choose file)"] + list(brand_map.values()))
    selected_file = None
    if selected_brand != "(choose file)":
        selected_file = [file for file, brand in brand_map.items() if brand == selected_brand][0]

# Upload alternative
uploaded_file = st.file_uploader("Or upload a PDF pricelist", type=["pdf"])
if uploaded_file is not None:
    # save to a temp buffer so other extraction functions can read from path-like as well
    uploaded_bytes = uploaded_file.read()
    # create BytesIO and pass into tabula/pdfplumber - pdfplumber supports file-like; tabula needs path on disk
    uploaded_buffer = io.BytesIO(uploaded_bytes)
    # write to disk to allow tabula to use it (if present)
    tmp_path = os.path.join(".", f"uploaded_{uploaded_file.name}")
    with open(tmp_path, "wb") as f:
        f.write(uploaded_bytes)
    selected_file = tmp_path
    st.success(f"Uploaded file saved as {tmp_path}")

# Nothing selected
if not selected_file:
    st.stop()

# -----------------------
# Try to extract table(s)
# -----------------------
st.write(f"Reading PDF: **{os.path.basename(selected_file)}**")

df = None
errors = []

# Prefer tabula if available (better table extraction for complex PDFs)
tabula = try_import("tabula")
if tabula:
    try:
        df = extract_tables_with_tabula(selected_file)
        st.info("Tables extracted with tabula.")
    except Exception as e:
        errors.append(f"tabula extraction failed: {e}")

# If tabula not available or failed, try pdfplumber
if df is None:
    try:
        df = extract_tables_with_pdfplumber(selected_file)
        st.info("Tables extracted with pdfplumber.")
    except Exception as e:
        errors.append(f"pdfplumber extraction failed: {e}")

if df is None:
    st.error("Could not extract tables from PDF. Install `tabula-py` (with Java) or `pdfplumber` and try again.")
    st.write("Errors encountered:")
    for e in errors:
        st.write("-", e)
    st.stop()

# Show raw extracted preview
st.write("#### Raw extracted table preview (first 10 rows)")
# it's possible columns are integers (0,1,2...), show them to the user
st.dataframe(df.head(10), use_container_width=True)

# Let user choose header row index (some PDFs put headers as 2nd row)
header_row = st.number_input("Header row index (0 = first extracted row). Set to 1 if your header is second row.", value=0, min_value=0, step=1)

# Apply header if that row exists
if header_row >= len(df):
    st.warning("Header row index is beyond the number of rows. Using row 0 as header.")
    header_row = 0

# If header_row > 0, convert the chosen row to header
if header_row > 0:
    new_header = df.iloc[header_row].fillna("").astype(str).tolist()
    df = df.drop(index=list(range(0, header_row + 1))).reset_index(drop=True)
    df.columns = new_header
else:
    # ensure sensible str column names
    df.columns = [str(c) for c in df.columns]

st.write("#### Table after applying header")
st.dataframe(df.head(10), use_container_width=True)

# -----------------------
# Detect numeric columns (candidates for rate)
# -----------------------
numeric_candidates = detect_numeric_columns(df)
st.write("Detected numeric-like columns (candidates for Rate):")
if numeric_candidates:
    st.write(", ".join([str(c) for c in numeric_candidates]))
else:
    st.write("_No obvious numeric columns detected — you can pick any column and we'll try to parse numbers from it._")

# Let user pick the rate column (or allow them to type)
all_columns = list(df.columns)
rate_col = st.selectbox("Choose Rate / Price column (the numeric column to use for calculation)", options=["(auto)"] + all_columns)

if rate_col == "(auto)":
    # pick first numeric candidate, else try heuristics for names like 'rate', 'price', 'mrp'
    chosen = None
    heuristics = ["rate", "price", "mrp", "selling", "net", "amount"]
    for h in heuristics:
        for c in all_columns:
            if h.lower() in str(c).lower():
                chosen = c
                break
        if chosen:
            break
    if not chosen and numeric_candidates:
        chosen = numeric_candidates[0]
    if not chosen:
        st.warning("Could not auto-detect a numeric rate column. Please select one manually.")
        st.stop()
    rate_col = chosen
    st.success(f"Auto-selected column: **{rate_col}**")

# Prepare numeric series
try:
    numeric_series = sanitize_numeric_series(df[rate_col])
    # convert to float; errors -> NaN
    numeric_series = pd.to_numeric(numeric_series, errors="coerce")
except Exception as e:
    st.error(f"Failed to coerce selected column ({rate_col}) to numeric: {e}")
    st.stop()

# Show some stats
st.write(f"Values parsed from **{rate_col}** (first 10):")
st.dataframe(pd.DataFrame({rate_col: numeric_series.head(10)}), use_container_width=True)

# Confirm calculation formula with the user and compute
st.write("### Calculation settings")
st.write("Formula used: `new_price = round(rate * 1.05 * 2.5)`")
multiplier_1 = st.number_input("Multiplier 1 (e.g., 1.05 for 105%)", value=1.05, step=0.01, format="%.4f")
multiplier_2 = st.number_input("Multiplier 2 (e.g., 2.5)", value=2.5, step=0.1, format="%.4f")
round_to = st.number_input("Round to nearest integer (0 = integer). For decimals, set number of decimals", value=0, min_value=0)

# compute
computed = numeric_series * multiplier_1 * multiplier_2
if round_to == 0:
    computed = computed.round(0).astype("Int64")  # preserve NA with pandas nullable Int
else:
    computed = computed.round(int(round_to))

# Add new column to dataframe (keep original df intact by creating a copy view)
result_df = df.copy()
result_df[f"NewPrice ({multiplier_1}x * {multiplier_2}x)"] = computed

st.write("#### Sample of generated pricelist (first 15 rows)")
st.dataframe(result_df.head(15), use_container_width=True)

# Search functionality (retained from your original)
query = st.text_input("Search for an item in generated pricelist (leave empty to show all)")
if query:
    filtered_df = result_df[result_df.apply(
        lambda row: row.astype(str).str.contains(query, case=False, na=False).any(),
        axis=1
    )]
    st.write(f"Showing results for **{query}**:")
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)
else:
    st.write("Showing full generated pricelist (first 200 rows):")
    st.dataframe(result_df.head(200), use_container_width=True, hide_index=True)

# Download button
csv_buf = result_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download generated pricelist as CSV",
    data=csv_buf,
    file_name=f"generated_pricelist_{os.path.splitext(os.path.basename(selected_file))[0]}.csv",
    mime="text/csv"
)

st.success("Done — generated pricelist is ready to download.")
