# streamlit_app_pricelist_pipeline_general_text_parser.py
import io
import os
import re
import csv
import json
import math
from datetime import datetime
from typing import Tuple, Optional, List
import pandas as pd
import streamlit as st

# ------------------ HARD-CODED CREDENTIALS / SETTINGS ------------------
GEMINI_API_KEY = "AIzaSyC1wkXBNq0c_YUaZy0fKtG8_vHgwsN2CoU"
MONGO_URI = "mongodb+srv://kunal-qd:Password_5202@cluster0.zem6dyp.mongodb.net/?appName=Cluster0"
MONGO_DBNAME = "pricelist_db"
MONGO_COLLECTION = "pricelist_final"

PREFERRED_MODEL_DEFAULT = "gemini-1.5-mini"
FALLBACK_MODELS_DEFAULT = "gemini-2.5-flash,gemini-2.5-pro,gemini-2.1,gemini-1.5-flash"
DEFAULT_MRP_FORMULA = "CUT * 2.5 * 1.05"
# -----------------------------------------------------------------------------------

# Optional libs
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except Exception:
    PDFPLUMBER_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.units import mm
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

try:
    from pymongo import MongoClient
    PYMONGO_AVAILABLE = True
except Exception:
    PYMONGO_AVAILABLE = False

def has_ocr_libs():
    try:
        import pdf2image, pytesseract  # noqa: F401
        return True
    except Exception:
        return False

# ---------------- Streamlit setup ----------------
st.set_page_config(page_title="Pricelist Pipeline", layout="wide")
st.title("Pricelist Pipeline")

# front page viewer default
view_mode = st.radio("Choose View", options=["Pricelist Viewer", "Pricelist Pipeline"], index=0, horizontal=True)

BRAND_OPTIONS = [
    "D3", "Purple Maze", "Maverick", "Fine Decor", "Balaji Decor",
    "D Decor", "G7 Blinds", "Balaji Blinds", "PL Decor", "Shivana"
]

# ---------------- Helpers ----------------
def numeric_from_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str)
         .str.replace(",", "", regex=False)
         .str.replace(" ", "", regex=False)
         .str.replace("Rs.", "", flags=re.IGNORECASE, regex=True)
         .str.replace("₹", "", regex=False),
        errors="coerce"
    )

# --- Gemini low-level caller (tries candidate models) ---
def gemini_generate_raw(prompt_text: str, preferred_model: str = None, fallback_models: list = None) -> Tuple[str, str]:
    if not GENAI_AVAILABLE:
        raise RuntimeError("google.generativeai not installed.")
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        pass
    tried = []
    candidates = []
    if preferred_model:
        candidates.append(preferred_model)
    if fallback_models:
        candidates.extend([m for m in fallback_models if m not in candidates])
    if not candidates:
        candidates = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.1", "gemini-1.5-mini", "gemini-1.5-flash"]
    last_exc = None
    for model_name in candidates:
        tried.append(model_name)
        try:
            model = genai.GenerativeModel(model_name)
            if hasattr(model, "generate_content"):
                resp = model.generate_content(prompt_text)
                text_out = getattr(resp, "text", None) or (resp.text if hasattr(resp, "text") else None)
            else:
                resp = genai.generate_text(model=model_name, prompt=prompt_text)
                text_out = resp.text if hasattr(resp, "text") else resp
            if not text_out:
                continue
            return text_out.strip(), model_name
        except Exception as e:
            last_exc = e
            continue
    raise RuntimeError(f"Gemini failed. Tried: {tried}. Last error: {last_exc}")

# ---------- tolerant CSV/JSON parsing helpers ----------
def try_parse_csv_text_to_df(text: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    text = (text or "").strip()
    if not text:
        return None, None
    try:
        df = pd.read_csv(io.StringIO(text), engine="python")
        return df, "pandas_csv_engine_python"
    except Exception:
        pass
    for d in [",", "\t", ";", "|"]:
        try:
            df = pd.read_csv(io.StringIO(text), sep=d, engine="python")
            if df.shape[1] >= 2 and df.shape[0] > 0:
                return df, f"pandas_sep_{d}"
        except Exception:
            pass
    try:
        sample = "\n".join(text.splitlines()[:20])
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample)
        reader = csv.reader(io.StringIO(text), dialect)
        rows = list(reader)
        if rows:
            maxcols = max(len(r) for r in rows)
            normalized = [r + [""] * (maxcols - len(r)) for r in rows]
            headers = [c.strip() if c.strip() else f"col_{i}" for i, c in enumerate(normalized[0])]
            df = pd.DataFrame(normalized[1:], columns=headers)
            return df, "csv_sniffer"
    except Exception:
        pass
    md_lines = [ln for ln in text.splitlines() if "|" in ln]
    if len(md_lines) >= 2:
        try:
            header_line = md_lines[0]
            headers = [h.strip() for h in re.split(r"\|", header_line) if h.strip() != ""]
            rows = []
            for ln in md_lines[1:]:
                cells = [c.strip() for c in re.split(r"\|", ln) if c.strip() != ""]
                if len(cells) == len(headers):
                    rows.append(cells)
            if rows:
                df = pd.DataFrame(rows, columns=headers)
                return df, "markdown_table"
        except Exception:
            pass
    lines = text.splitlines()
    try:
        objs = []
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            if ln.startswith("{") and ln.endswith("}"):
                objs.append(json.loads(ln))
            else:
                objs = []
                break
        if objs:
            df = pd.DataFrame(objs)
            return df, "ndjson"
    except Exception:
        pass
    try:
        data = json.loads(text)
        if isinstance(data, list):
            df = pd.DataFrame(data)
            return df, "json_array"
        elif isinstance(data, dict):
            df = pd.json_normalize(data)
            return df, "json_object"
    except Exception:
        pass
    return None, None

# ---------- pdfplumber table extractor ----------
def try_pdfplumber_tables(file_bytes: bytes) -> Optional[pd.DataFrame]:
    if not PDFPLUMBER_AVAILABLE:
        return None
    try:
        tables = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for pagenum, page in enumerate(pdf.pages):
                try:
                    tbl = page.extract_table()
                except Exception:
                    tbl = None
                if tbl and len(tbl) > 1:
                    header = [str(h).strip() if h is not None else f"col_{i}" for i, h in enumerate(tbl[0])]
                    rows = tbl[1:]
                    df_page = pd.DataFrame(rows, columns=header)
                    df_page['__source_page'] = pagenum + 1
                    tables.append(df_page)
                else:
                    try:
                        tbs = page.extract_tables()
                        for t in (tbs or []):
                            if len(t) > 1:
                                header = [str(h).strip() if h is not None else f"col_{i}" for i, h in enumerate(t[0])]
                                rows = t[1:]
                                df_page = pd.DataFrame(rows, columns=header)
                                df_page['__source_page'] = pagenum + 1
                                tables.append(df_page)
                    except Exception:
                        pass
        if not tables:
            return None
        max_cols = max(t.shape[1] for t in tables)
        normalized = []
        for t in tables:
            if t.shape[1] < max_cols:
                for i in range(max_cols - t.shape[1]):
                    t[f"_extra_{i}"] = ""
            normalized.append(t)
        big = pd.concat(normalized, ignore_index=True, sort=False)
        big = big[[c for c in big.columns if big[c].astype(str).str.strip().ne("").any()]]
        return big.reset_index(drop=True)
    except Exception:
        return None

# ---------- postprocess pdfplumber wide tables (kept as fallback) ----------
def postprocess_pdfplumber_df(df_raw: pd.DataFrame, min_numeric_ratio=0.5) -> Optional[pd.DataFrame]:
    if df_raw is None or df_raw.empty:
        return None
    df = df_raw.copy()
    df = df.rename(columns=lambda c: str(c).strip() if c is not None else "")
    for c in df.columns:
        df[c] = df[c].astype(str).replace("nan", "").replace("None", "").str.strip()
    non_empty_cols = [c for c in df.columns if df[c].astype(bool).any()]
    df = df[non_empty_cols].copy()
    if df.shape[1] == 0:
        return None
    if df.shape[1] > df.shape[0] and df.shape[0] <= 10:
        try:
            df_t = df.transpose().reset_index(drop=True)
            for c in df_t.columns:
                df_t[c] = df_t[c].astype(str).str.strip()
            if df_t.shape[0] > df.shape[0]:
                df = df_t
        except Exception:
            pass
    def numeric_score(series):
        s = series.astype(str).str.replace(r"[^\d.\-]", "", regex=True)
        valid = pd.to_numeric(s.replace("", pd.NA), errors="coerce").notna().sum()
        return valid / max(1, len(series))
    col_scores = {c: numeric_score(df[c]) for c in df.columns}
    numeric_cols = [c for c, sc in col_scores.items() if sc >= min_numeric_ratio]
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]
    if not numeric_cols:
        numeric_candidates = sorted(col_scores.items(), key=lambda x: x[1], reverse=True)
        numeric_cols = [c for c, sc in numeric_candidates if sc > 0][:2]
        non_numeric_cols = [c for c in df.columns if c not in numeric_cols]
    cols_list = list(df.columns)
    lead_non_numeric = []
    for c in cols_list:
        if c in non_numeric_cols:
            lead_non_numeric.append(c)
        else:
            break
    if not lead_non_numeric:
        lead_non_numeric = non_numeric_cols
    def merge_catalogue_row(row):
        parts = [str(row[c]).strip() for c in lead_non_numeric if str(row[c]).strip()]
        return " | ".join(parts).strip()
    df["Catalogue"] = df.apply(merge_catalogue_row, axis=1)
    try:
        last_text_col_idx = max([df.columns.get_loc(c) for c in lead_non_numeric]) if lead_non_numeric else -1
    except Exception:
        last_text_col_idx = -1
    cut_cols = []
    for i, c in enumerate(df.columns):
        if i > last_text_col_idx and c in numeric_cols:
            cut_cols.append(c)
    if not cut_cols:
        cut_cols = numeric_cols[:2]
    for idx, c in enumerate(cut_cols):
        out_name = f"CUT_{idx+1}"
        df[out_name] = pd.to_numeric(df[c].astype(str).str.replace(r"[^\d.\-]", "", regex=True).replace("", pd.NA), errors="coerce")
    if not any(str(col).startswith("CUT_") for col in df.columns):
        def last_numeric_token(row):
            tokens = " ".join([str(v) for v in row]).split()
            nums = [re.sub(r"[^\d.\-]", "", t) for t in tokens]
            nums = [n for n in nums if n and re.search(r"\d", n)]
            return float(nums[-1]) if nums else None
        df["CUT_1"] = df.apply(last_numeric_token, axis=1)
    keep_cols = ["Catalogue"] + [c for c in df.columns if str(c).startswith("CUT_")]
    df_clean = df[keep_cols].copy()
    df_clean = df_clean[~(df_clean["Catalogue"].astype(str).str.strip() == "") | (df_clean[[c for c in df_clean.columns if c.startswith("CUT_")]].notna().any(axis=1))].reset_index(drop=True)
    header_noise = df_clean["Catalogue"].astype(str).str.lower().str.contains(r"rate|cut|rs|catalog|code|width|gst|mrp|price")
    if header_noise.any():
        df_clean = df_clean[~header_noise].reset_index(drop=True)
    if df_clean["Catalogue"].astype(str).str.strip().replace("", pd.NA).notna().sum() == 0:
        first_col = df.columns[0]
        df_clean["Catalogue"] = df[first_col].astype(str).fillna("")
    df_clean["Catalogue"] = df_clean["Catalogue"].astype(str).str.strip()
    for c in df_clean.columns:
        if c.startswith("CUT_"):
            df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce")
    return df_clean.reset_index(drop=True)

# ---------- text-first general parser (NEW) ----------
def parse_structured_text_rows(text: str) -> Optional[pd.DataFrame]:
    """
    General text-based parser for line-oriented price lists.
    Heuristics:
      - Split text into lines, collapse repeated headers.
      - Identify contiguous 'record' lines: those containing >= 2 numeric tokens (rates).
      - For multi-line catalogue names, merge lines until a numeric token appears.
      - Extract tokens using regex: NAME then numbers: CUT, ROLL, WIDTH (like 54"), HSN, GST, optional FILE_COST.
    Returns DataFrame with columns:
      Catalogue, CUT_RATE, ROLL_RATE, WIDTH, HSN_CODE, GST, FILE_COST (where available)
    """
    if not text:
        return None
    # normalize whitespace and replace Unicode non-breaking spaces
    text = text.replace("\xa0", " ").replace("\u200b", "")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None

    # remove obvious repeated header lines (heuristic)
    header_candidates = []
    cleaned_lines = []
    for ln in lines:
        low = ln.lower()
        if any(h in low and len(low.split()) < 8 for h in ("cut", "roll", "rate", "hsn", "gst", "width", "catalog")):
            # treat as header line — store for possible use but skip in records
            header_candidates.append(ln)
            continue
        cleaned_lines.append(ln)

    # merge broken lines: when a line does NOT contain any numeric token but next line does,
    # it's likely part of the catalogue name -> append to next line
    merged_lines = []
    buffer = None
    numeric_re = re.compile(r"(?<!\w)(?:\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.\d+|\d+)(?:\"|''|”)?")  # matches numbers and numbers with "
    for ln in cleaned_lines:
        if buffer is None:
            buffer = ln
        else:
            # if buffer already has numeric tokens, push it and start new
            if numeric_re.search(buffer):
                merged_lines.append(buffer)
                buffer = ln
            else:
                # buffer has no numbers; merge with current line (part of name)
                buffer = buffer + " " + ln
    if buffer:
        merged_lines.append(buffer)

    # Now, consider each merged_line: lines that contain >=2 numeric tokens are records
    records = []
    for ln in merged_lines:
        nums = numeric_re.findall(ln)
        if len(nums) >= 2:
            # attempt structured extraction:
            # pattern: <name> <num1> <num2> <width?> <hsn?> <gst%?> <optional filecost?>
            # We'll split tokens while keeping quoted widths like 54"
            # First, find numeric tokens with their spans to separate name from numbers
            num_spans = [(m.group(0), m.start(), m.end()) for m in numeric_re.finditer(ln)]
            first_num_pos = num_spans[0][1]
            name_token = ln[:first_num_pos].strip(" -|:,.")
            # tokens after name
            tail = ln[first_num_pos:].strip()
            # split tail by whitespace but keep width tokens (like 54")
            tail_parts = re.split(r"\s+", tail)
            # normalize parts: remove commas from numbers
            def norm_num(tok):
                return tok.replace(",", "").strip().strip('"').strip("'").strip("”")
            # collect candidates
            # heuristics: first numeric = CUT, second = ROLL, third maybe width (contains "), maybe HSN (6 digits), GST contains %
            cut = roll = width = hsn = gst = filecost = None
            # find numeric-like tokens among tail_parts in order (keep tokens with digits)
            digit_parts = [p for p in tail_parts if re.search(r"\d", p)]
            if digit_parts:
                # assign
                if len(digit_parts) >= 1:
                    cut = norm_num(digit_parts[0])
                if len(digit_parts) >= 2:
                    roll = norm_num(digit_parts[1])
                # search remaining tokens for width (contains ")
                for p in digit_parts[2:]:
                    if '"' in p or "inch" in p.lower() or re.search(r'\d{2}"', p):
                        width = p.replace("”", '"').strip()
                        continue
                    # HSN: often 6 digits
                    if re.fullmatch(r"\d{6}", re.sub(r"[^\d]", "", p)):
                        hsn = re.sub(r"[^\d]", "", p)
                        continue
                    # GST: contains %
                    if "%" in p:
                        gst = p.strip()
                        continue
                # optional filecost may be trailing numeric > 3 digits
                if len(digit_parts) >= 3:
                    # heuristic: if there are more numeric tokens beyond the first two, pick last numeric as possible file cost if it looks large
                    last_num = norm_num(digit_parts[-1])
                    try:
                        if last_num and float(last_num) > 1000:
                            filecost = last_num
                    except Exception:
                        pass
            # finalize record
            record = {
                "Catalogue": name_token if name_token else ln,
                "CUT_RATE": try_parse_float_or_none(cut),
                "ROLL_RATE": try_parse_float_or_none(roll),
                "WIDTH": normalize_width(width),
                "HSN_CODE": hsn,
                "GST": normalize_gst(gst),
                "FILE_COST": try_parse_float_or_none(filecost)
            }
            records.append(record)
        else:
            # not enough numeric tokens - skip or treat as catalog-only row (rare)
            # we'll append a record with only Catalogue if beneficial (skip for now)
            pass

    if not records:
        return None
    df = pd.DataFrame(records)
    # clean columns: ensure Catalogues stripped and unique rows kept
    df["Catalogue"] = df["Catalogue"].astype(str).str.replace(r"\s{2,}", " ", regex=True).str.strip()
    # drop empty rows (no catalogue and no numeric)
    df = df[~((df["Catalogue"].astype(str).str.strip() == "") & (df[["CUT_RATE", "ROLL_RATE"]].isna().all(axis=1)))]
    if df.empty:
        return None
    # sometimes CUT_RATE is missing but ROLL_RATE present - fill CUT with ROLL in that case
    df["CUT_RATE"] = df.apply(lambda r: r["ROLL_RATE"] if pd.isna(r["CUT_RATE"]) and not pd.isna(r["ROLL_RATE"]) else r["CUT_RATE"], axis=1)
    # reset index
    return df.reset_index(drop=True)

# small helpers used by text parser
def try_parse_float_or_none(s):
    if s is None:
        return None
    try:
        s2 = str(s).replace(",", "").strip()
        if s2 == "":
            return None
        return float(re.sub(r"[^\d.\-]", "", s2))
    except Exception:
        return None

def normalize_width(s):
    if not s:
        return None
    s = str(s).strip().replace("”", '"')
    # keep forms like 54", 48", 2.5m etc.
    m = re.search(r'(\d{1,3}(?:\.\d+)?\s*("|in|inch|inches)?)', s, flags=re.I)
    if m:
        return m.group(1).replace(" ", "")
    # fallback: digits
    m2 = re.search(r'(\d{1,3}(?:\.\d+)?)', s)
    if m2:
        return m2.group(1)
    return s

def normalize_gst(s):
    if not s:
        return None
    s = str(s).strip()
    if "%" in s:
        return s
    # maybe '5' -> '5%'
    m = re.search(r'\d+(\.\d+)?', s)
    if m:
        return m.group(0) + "%"
    return s

# ---------- Gemini JSON / OCR helpers (kept) ----------
def gemini_json_parse_from_text(text: str, preferred_model: str=None, fallback_models: list=None) -> Tuple[Optional[pd.DataFrame], str, str]:
    json_prompt = f"""
You are a JSON-only parser for vendor price lists. Convert the RAW TEXT into a JSON array ONLY.
Each element must be an object with keys:
"SNO", "CATALOGUE", "CODE", "CUT_RATE", "GST", "WIDTH"
- If a field is missing, use "" or null.
- Merge multi-line product names into CATALOGUE.
- Remove currency symbols and commas from numeric fields.
- Return VALID JSON only. No explanation text.
RAW TEXT:
{text}
"""
    try:
        raw_text, model_used = gemini_generate_raw(json_prompt, preferred_model=preferred_model, fallback_models=fallback_models)
    except Exception as e:
        return None, f"gemini_call_failed:{e}", ""
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, list):
            df = pd.DataFrame(parsed)
            return df, "gemini_json", raw_text
    except Exception:
        df_tab, method = try_parse_csv_text_to_df(raw_text)
        if df_tab is not None:
            return df_tab, "gemini_raw_parsed_table", raw_text
        return None, "gemini_json_parse_failed", raw_text

def ocr_pdf_to_df(file_bytes: bytes) -> Optional[pd.DataFrame]:
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
    except Exception:
        return None
    try:
        images = convert_from_bytes(file_bytes, dpi=200)
    except Exception:
        return None
    ocr_texts = []
    for img in images:
        txt = pytesseract.image_to_string(img, lang='eng')
        ocr_texts.append(txt)
    bigtext = "\n\n".join(ocr_texts)
    df_tab, method = try_parse_csv_text_to_df(bigtext)
    if df_tab is not None:
        return df_tab
    lines = [ln for ln in bigtext.splitlines() if ln.strip()]
    rows = []
    for ln in lines:
        parts = re.split(r"\s{2,}|\t", ln.strip())
        if len(parts) >= 2:
            rows.append(parts)
    if rows:
        maxc = max(len(r) for r in rows)
        normalized = [r + [""]*(maxc - len(r)) for r in rows]
        headers = [f"col_{i}" for i in range(maxc)]
        return pd.DataFrame(normalized, columns=headers)
    return None

# ---------- Top-level robust parse (text-first) ----------
def robust_parse_pdf(file_bytes: bytes, preferred_model: str=None, fallback_models: list=None, allow_ocr: bool=False) -> Tuple[Optional[pd.DataFrame], str, Optional[str]]:
    """
    Parsing order (text-first general approach):
      1) extract text (pdfplumber) -> run parse_structured_text_rows(text)
      2) if text parser fails, try pdfplumber table extractor + postprocess
      3) gemini json parse of extracted text
      4) optional OCR fallback
    Returns (df_or_none, method_str, debug_raw_text_if_any)
    """
    extracted_text = None
    # 1) extract text via pdfplumber if available
    if PDFPLUMBER_AVAILABLE:
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                pages_text = [p.extract_text() or "" for p in pdf.pages]
            extracted_text = "\n\n".join(pages_text).strip()
        except Exception:
            extracted_text = None

    # If we have extracted text, run the general text parser first
    if extracted_text:
        df_text = parse_structured_text_rows(extracted_text)
        if df_text is not None and not df_text.empty:
            return df_text, "text_parser_structured_rows", extracted_text

    # 2) try pdfplumber tables + postprocess (fallback)
    df_pdf = try_pdfplumber_tables(file_bytes)
    if df_pdf is not None:
        df_clean = postprocess_pdfplumber_df(df_pdf)
        if df_clean is not None and not df_clean.empty:
            return df_clean, "pdfplumber_tables_postprocessed", None

    # 3) try Gemini JSON on extracted text (if allowed / available)
    if extracted_text and GENAI_AVAILABLE:
        df_gem, method_or_msg, raw_text = gemini_json_parse_from_text(extracted_text, preferred_model=preferred_model, fallback_models=fallback_models)
        if df_gem is not None:
            return df_gem, method_or_msg, raw_text
        else:
            # if Gemini failed and OCR not enabled, return failure with gemini raw text for debugging
            if not allow_ocr:
                return None, method_or_msg, raw_text

    # 4) OCR fallback
    if allow_ocr and has_ocr_libs():
        df_ocr = ocr_pdf_to_df(file_bytes)
        if df_ocr is not None:
            return df_ocr, "ocr_fallback", None

    return None, "no_parse_succeeded", extracted_text

# ---------------- MongoDB client ----------------
mongo_client = None
if PYMONGO_AVAILABLE and MONGO_URI:
    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        mongo_client.admin.command("ping")
    except Exception:
        mongo_client = None

# ---------------- Viewer (front page) ----------------
if view_mode == "Pricelist Viewer":
    st.header("Pricelist Viewer")
    if not PYMONGO_AVAILABLE:
        st.warning("pymongo not installed — viewer requires pymongo.")
        st.stop()
    if not mongo_client:
        st.error("MongoDB client not available. Check MONGO_URI.")
        st.stop()
    db = mongo_client[MONGO_DBNAME]
    coll = db[MONGO_COLLECTION]
    show_only_with_data = st.checkbox("Show only brands with saved pricelists", value=False)
    brands_to_show = BRAND_OPTIONS
    if show_only_with_data:
        available = []
        for b in BRAND_OPTIONS:
            if coll.find_one({"brand": b, "pricelist": {"$exists": True, "$ne": []}}):
                available.append(b)
        brands_to_show = available or BRAND_OPTIONS
    brand = st.selectbox("Select Brand", ["-- select brand --"] + brands_to_show)
    if brand == "-- select brand --":
        st.info("Select a brand to view its pricelist.")
        st.stop()
    doc = coll.find_one({"brand": brand})
    if not doc or "pricelist" not in doc or not doc["pricelist"]:
        st.info(f"No pricelist found for brand '{brand}'.")
        st.stop()
    pricelist = doc["pricelist"]
    rows = []
    all_value_keys = set()
    for item in pricelist:
        values = item.get("values", {}) or {}
        all_value_keys.update(values.keys())
    mrp_keys = sorted([k for k in all_value_keys if str(k).startswith("MRP_")])
    if not mrp_keys:
        st.info("No MRP columns found for this brand — showing available saved columns.")
        ordered_keys = sorted(list(all_value_keys))
    else:
        ordered_keys = mrp_keys
    for item in pricelist:
        row = {"Catalogue": item.get("model", "")}
        values = item.get("values", {}) or {}
        for k in ordered_keys:
            row[k] = values.get(k, None)
        rows.append(row)
    df_view = pd.DataFrame(rows)
    if df_view.empty:
        st.info("No rows to display.")
        st.stop()
    df_view.insert(0, "SNO", range(1, len(df_view) + 1))
    search_q = st.text_input("Search in catalogue (case-insensitive substring). Leave blank to show all.")
    if search_q:
        mask = df_view["Catalogue"].astype(str).str.contains(search_q, case=False, na=False)
        df_filtered = df_view[mask].reset_index(drop=True)
        st.write(f"Showing {len(df_filtered)} / {len(df_view)} rows matching '{search_q}'")
        st.dataframe(df_filtered.set_index("SNO"), use_container_width=True)
        st.download_button("Download visible pricelist as CSV", df_filtered.to_csv(index=False).encode("utf-8"), file_name=f"{brand}_filtered.csv", mime="text/csv")
    else:
        st.dataframe(df_view.set_index("SNO"), use_container_width=True)
        st.download_button("Download pricelist as CSV", df_view.to_csv(index=False).encode("utf-8"), file_name=f"{brand}_pricelist.csv", mime="text/csv")
    created = doc.get("created_date")
    updated = doc.get("updated_last")
    if created or updated:
        st.markdown("---")
        if created:
            st.write("Created:", created)
        if updated:
            st.write("Updated last:", updated)
    st.stop()

# ---------------- Pricelist Pipeline (separate view) ----------------
st.header("Pricelist Pipeline")
col1, col2 = st.columns([2, 1])
with col1:
    preferred_model = st.text_input("Gemini preferred model", value=PREFERRED_MODEL_DEFAULT)
    fallback_models_input = st.text_input("Gemini fallback models (comma-separated)", value=FALLBACK_MODELS_DEFAULT)
    fallback_models = [m.strip() for m in fallback_models_input.split(",") if m.strip()]
    use_gemini = st.checkbox("Enable Gemini parsing", value=True)
with col2:
    mrp_formula = st.text_input("MRP formula (use CUT)", value=DEFAULT_MRP_FORMULA)
    allow_ocr = st.checkbox("Enable OCR fallback (slow; requires pdf2image & pytesseract)", value=False)

st.header("Upload Pricelist")
uploaded = st.file_uploader("Upload vendor pricelist (PDF / CSV / Excel)", type=["pdf", "csv", "xls", "xlsx"])
if not uploaded:
    st.info("Upload a file to begin.")
    st.stop()

filename = uploaded.name
file_bytes = uploaded.read()
extracted_text = None
prelim_df = None
if filename.lower().endswith(".pdf"):
    if not PDFPLUMBER_AVAILABLE:
        st.warning("pdfplumber not installed — recommended for PDF extraction.")
    else:
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                pages_text = [p.extract_text() or "" for p in pdf.pages]
            extracted_text = "\n\n".join(pages_text).strip()
        except Exception:
            extracted_text = None
else:
    if filename.lower().endswith(".csv"):
        prelim_df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        sheets = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
        if len(sheets) == 1:
            prelim_df = list(sheets.values())[0]
        else:
            chosen = st.selectbox("Multiple sheets found — select one", list(sheets.keys()))
            prelim_df = sheets[chosen]

st.header("Parse / Clean")
st.write("Parsing flow: text-first parser → pdfplumber tables postprocess → Gemini JSON → optional OCR fallback.")
parse_btn = st.button("Parse (robust)")

if parse_btn:
    with st.spinner("Parsing..."):
        df_parsed, method, debug_raw = robust_parse_pdf(file_bytes, preferred_model=preferred_model if use_gemini else None, fallback_models=fallback_models, allow_ocr=allow_ocr)
    if df_parsed is None:
        st.error(f"Parsing failed (method: {method}). See debug output below.")
        if debug_raw:
            st.subheader("Extracted raw text (first 20000 chars)")
            st.text_area("Extracted raw", debug_raw[:20000], height=300)
        else:
            st.info("No extracted raw text available. Try enabling OCR fallback or check pdfplumber installation.")
        st.stop()
    else:
        st.success(f"Parsed using: {method}")
        st.dataframe(df_parsed.head(200), use_container_width=True)
        st.session_state['parsed_df'] = df_parsed.copy()
        st.session_state['parse_method'] = method

if 'parsed_df' not in st.session_state and prelim_df is not None:
    st.info("Using table loaded directly from uploaded file.")
    st.session_state['parsed_df'] = prelim_df.copy()

if 'parsed_df' not in st.session_state:
    st.info("Parsed table not available yet. Click Parse (robust) or upload CSV/XLSX.")
    st.stop()

# Map columns & calculate MRP
df = st.session_state['parsed_df'].copy()
st.header("Map Columns & Calculate MRP")
st.write("Select Catalogue/Model column and one or more numeric columns (CUT) to compute MRP.")
df.columns = [str(c).strip() for c in df.columns]

def guess_name_cols(df_):
    keys = ["model", "name", "catalog", "catalogue", "desc", "description", "item", "product"]
    return [c for c in df_.columns if any(re.search(rf"\b{kw}\b", str(c), flags=re.I) for kw in keys)]

def guess_numeric_cols(df_):
    numeric_scores = []
    for c in df_.columns:
        try:
            n = numeric_from_series(df_[c]).count()
            numeric_scores.append((c, n))
        except Exception:
            numeric_scores.append((c, 0))
    numeric_scores.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in numeric_scores if _ > 0]

name_candidates = guess_name_cols(df)
numeric_candidates = guess_numeric_cols(df)

catalogue_col = st.selectbox("Catalogue / Model column", options=(name_candidates + [c for c in df.columns if c not in name_candidates]))
cut_cols = st.multiselect("Select one or more numeric CUT columns", options=(numeric_candidates + [c for c in df.columns if c not in numeric_candidates]))
if not cut_cols:
    st.warning("Select at least one numeric column to compute MRP.")
    st.stop()

df_work = df.copy()
for cut_col in cut_cols:
    df_work[f"_CUTNUM__{cut_col}"] = numeric_from_series(df_work[cut_col])

def safe_eval_mrp(cut_value, formula_text):
    safe_locals = {"CUT": cut_value, "math": math, "round": round, "int": int}
    try:
        return eval(formula_text, {"__builtins__": {}}, safe_locals)
    except Exception:
        return None

for cut_col in cut_cols:
    df_work[f"MRP_{cut_col}"] = df_work[f"_CUTNUM__{cut_col}"].apply(lambda x: safe_eval_mrp(x, mrp_formula) if pd.notna(x) else None)

preview = pd.DataFrame({"Catalogue": df_work[catalogue_col].astype(str).fillna("")})
for c in cut_cols:
    preview[f"MRP_{c}"] = df_work[f"MRP_{c}"]

preview.insert(0, "SNO", range(1, len(preview) + 1))

def normalize_mrp(v):
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return int(float(v))
    except Exception:
        return None

for c in cut_cols:
    preview[f"MRP_{c}"] = preview[f"MRP_{c}"].apply(normalize_mrp)

st.header("Preview (Catalogue + MRP columns)")
st.dataframe(preview.set_index("SNO"), use_container_width=True)

st.header("Save to MongoDB")
brand_sel = st.selectbox("Select Brand to save under", options=BRAND_OPTIONS)

if st.button("Save final pricelist to MongoDB (upsert by brand)"):
    if not PYMONGO_AVAILABLE or not mongo_client:
        st.error("MongoDB not available. Check pymongo and MONGO_URI.")
        st.stop()
    db = mongo_client[MONGO_DBNAME]
    coll = db[MONGO_COLLECTION]
    pricelist_docs = []
    for idx, row in preview.reset_index(drop=True).iterrows():
        model_name = row["Catalogue"]
        values = {}
        for c in cut_cols:
            orig_col = f"_CUTNUM__{c}"
            orig_val = None
            if orig_col in df_work.columns:
                v = df_work.loc[df_work.index[idx], orig_col]
                orig_val = None if pd.isna(v) else float(v)
            values[c] = (None if orig_val is None else float(orig_val))
            values[f"MRP_{c}"] = (None if pd.isna(row.get(f"MRP_{c}")) else int(row.get(f"MRP_{c}")))
        pricelist_docs.append({"model": model_name, "values": values})
    now = datetime.utcnow()
    existing = coll.find_one({"brand": brand_sel})
    if existing:
        try:
            coll.update_one({"brand": brand_sel}, {"$set": {"pricelist": pricelist_docs, "updated_last": now, "_source_file": filename}})
            st.success(f"Updated brand '{brand_sel}' with {len(pricelist_docs)} rows.")
        except Exception as e:
            st.error(f"Failed to update: {e}")
    else:
        doc = {"brand": brand_sel, "pricelist": pricelist_docs, "created_date": now, "updated_last": now, "_source_file": filename}
        try:
            res = coll.insert_one(doc)
            st.success(f"Inserted brand '{brand_sel}' (id: {res.inserted_id}). Rows: {len(pricelist_docs)}")
        except Exception as e:
            st.error(f"Failed to insert: {e}")

st.header("Download PDF (optional)")
if REPORTLAB_AVAILABLE:
    try:
        def to_pdf_bytes(df: pd.DataFrame, title: str = "Pricelist — MRP") -> bytes:
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
            from reportlab.lib.styles import ParagraphStyle
            from reportlab.lib.units import mm
            from reportlab.lib import colors
            buf = io.BytesIO()
            page_size = A4
            doc = SimpleDocTemplate(buf, pagesize=page_size, rightMargin=12*mm, leftMargin=12*mm, topMargin=12*mm, bottomMargin=12*mm)
            elements = []
            title_style = ParagraphStyle(name="Title", fontSize=14, leading=16, alignment=1)
            elements.append(Paragraph(title, title_style))
            elements.append(Spacer(1, 6))
            df_display = df.copy().fillna("")
            headers = list(df_display.columns.astype(str))
            data_rows = df_display.astype(str).values.tolist()
            usable_width = page_size[0] - doc.leftMargin - doc.rightMargin
            n = len(headers)
            if n >= 3:
                col_widths = [usable_width * 0.06, usable_width * 0.60] + [usable_width * 0.34 / max(1, n-2)] * (n-2)
            else:
                col_widths = [usable_width / n] * n
            rows_per_page = 40
            total_rows = len(data_rows)
            start = 0
            while start < total_rows:
                chunk = data_rows[start:start + rows_per_page]
                table_data = [headers] + chunk
                from reportlab.platypus import Table
                table = Table(table_data, colWidths=col_widths, repeatRows=1)
                style = TableStyle([
                    ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
                    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f2f2f2")),
                    ("ALIGN", (0,0), (-1,-1), "LEFT"),
                    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                    ("FONTSIZE", (0,0), (-1,-1), 9),
                    ("BOTTOMPADDING", (0,0), (-1, -1), 4),
                    ("TOPPADDING", (0,0), (-1, -1), 4),
                ])
                table.setStyle(style)
                elements.append(table)
                start += rows_per_page
                if start < total_rows:
                    elements.append(PageBreak())
            doc.build(elements)
            pdf_bytes = buf.getvalue()
            buf.close()
            return pdf_bytes
        pdf_bytes = to_pdf_bytes(preview, title=f"{os.path.splitext(filename)[0]} — Catalogue & MRPs")
        st.download_button("Download final PDF (SNO, Catalogue, MRPs)", pdf_bytes, file_name=f"{os.path.splitext(filename)[0]}_catalogue_MRPs.pdf", mime="application/pdf")
    except Exception as e:
        st.error(f"Failed to build PDF: {e}")
else:
    st.info("Install reportlab (pip install reportlab) to enable PDF export.")

if 'parse_method' in st.session_state:
    st.caption(f"Last parse method used: {st.session_state['parse_method']}")
