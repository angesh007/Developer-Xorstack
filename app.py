# app.py
import os
import io
import csv
import json
import zipfile
import tempfile
import re
from typing import Optional, List, Dict
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from openai import OpenAI
import dateparser

# optional spaCy fallback (used to augment OpenAI when needed)
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
API_KEY = os.getenv("API_KEY", None)

if not OPENAI_KEY:
    print("Warning: OPENAI_API_KEY not set in environment. Set it in .env or env var.")
client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None

CSV_FILE = "results.csv"
INGEST_BASE = os.path.join(os.getcwd(), "ingest")
os.makedirs(INGEST_BASE, exist_ok=True)

# ensure CSV header exists (add original_text column)
CSV_FIELDS = ["source_filename", "persons", "dates", "locations", "raw_json", "original_text"]
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

app = FastAPI(title="AI Extract Pipeline (robust parsing + merge fallback)")

# ---------------- helpers ----------------
def normalize_date_to_iso(date_text: str) -> Optional[str]:
    if not date_text:
        return None
    try:
        dt = dateparser.parse(date_text, settings={"RETURN_AS_TIMEZONE_AWARE": False})
        if dt:
            return dt.date().isoformat()
    except Exception:
        pass
    return None

def extract_json_blob(text: str) -> Optional[str]:
    """Return the first { ... } substring found in text, or None."""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    return None

def _safe_json_load(s: str):
    """Try json.loads but return (obj, None) or (None, error)."""
    try:
        return json.loads(s), None
    except Exception as e:
        return None, str(e)

def _clean_escaped_json(content: str) -> str:
    """
    Clean common escape patterns that models sometimes return.
    """
    c = content.strip()
    # strip wrapping quotes
    if (c.startswith('"') and c.endswith('"')) or (c.startswith("'") and c.endswith("'")):
        c = c[1:-1]
    # replace escaped quotes sequences
    c = re.sub(r'\\+"', '"', c)
    c = re.sub(r'"\\"+', '"', c)
    c = c.replace('\\n', '\n')
    c = re.sub(r'\\\\+', r'\\', c)
    c = c.replace('"\\"', '"').replace('\\""', '"')
    return c

def extract_with_spacy_and_regex(text: str) -> Dict:
    """Fallback extractor using spaCy NER (if available) and regex for dates."""
    persons = []
    dates = []
    locations = []
    if nlp:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                persons.append(ent.text.strip())
            elif ent.label_ == "DATE":
                dates.append(ent.text.strip())
            elif ent.label_ in ("GPE", "LOC", "FAC"):
                locations.append(ent.text.strip())
    # regex for dates (supplement)
    if not dates:
        date_candidates = re.findall(
            r'\b(?:\d{1,2}[\/.-]\d{1,2}[\/.-]\d{2,4}|\d{4}[\/.-]\d{1,2}[\/.-]\d{1,2}|[A-Za-z]{3,9}\s+\d{1,2},\s*\d{4}|[0-9]{1,2}\s+[A-Za-z]{3,9}\s+[0-9]{4})\b',
            text)
        dates.extend(date_candidates)
    # dedupe preserve order
    def dedupe(seq):
        seen = set(); out=[]
        for s in seq:
            s2 = s.strip()
            if not s2:
                continue
            if s2 not in seen:
                seen.add(s2); out.append(s2)
        return out
    persons = dedupe(persons)
    dates = dedupe(dates)
    locations = dedupe(locations)
    iso_dates = []
    for d in dates:
        iso = normalize_date_to_iso(d)
        iso_dates.append(iso if iso else d)
    return {"persons": persons, "dates": iso_dates, "locations": locations, "source": "spacy-regex"}

def _reformat_openai_output(raw_content: str, model: str = "gpt-3.5-turbo") -> Dict:
    """Ask the model to reformat a previous raw output into valid JSON only."""
    if not client:
        raise RuntimeError("OpenAI client not configured for reformatting")
    reformat_prompt = (
        "The assistant previously returned the following RAW text which was intended to be a JSON object:\n\n"
        "----BEGIN RAW----\n" + raw_content + "\n----END RAW----\n\n"
        "Please ONLY return a single valid JSON object (no surrounding quotes, no explanation) "
        "with keys: \"persons\" (array of person names), \"dates\" (array of ISO YYYY-MM-DD where possible or original strings), "
        "\"locations\" (array), and \"source\":\"openai\". Include single-token names like 'Ram' when present.\n\n"
        "If the RAW text is already valid JSON, just return that cleaned JSON."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": reformat_prompt}],
        temperature=0
    )
    try:
        content = resp.choices[0].message["content"]
    except Exception:
        content = str(resp)
    blob = extract_json_blob(content)
    if blob:
        obj, err = _safe_json_load(blob)
        if obj is not None:
            obj.setdefault("source", "openai")
            obj.setdefault("extraction_path", "openai-reformatted")
            return obj
    return {"error": "openai-reformat-failed", "raw": content, "source": "openai", "extraction_path": "openai-reformat-failed"}

def call_openai_extract(text: str, model: str = "gpt-3.5-turbo") -> Dict:
    """Call OpenAI with a tight prompt; try direct parse, cleaned parse, reformat. Return result dict."""
    if not client:
        raise RuntimeError("OpenAI client not configured")

    prompt = (
        "Extract PERSON names, DATES and LOCATIONS from the input text. "
        "Return ONLY a single valid JSON object (no explanation, no surrounding quotes or code fences) with keys:\n"
        "  - \"persons\": array of person names (include single-token names like 'Ram' if present),\n"
        "  - \"dates\": array of ISO-format strings (YYYY-MM-DD) when possible, otherwise original string,\n"
        "  - \"locations\": array of location names,\n"
        "  - \"source\": the string \"openai\".\n\n"
        "Do NOT escape quotes or wrap the JSON in another string. Example:\n"
        '{ "persons": ["John Doe"], "dates": ["2020-01-05"], "locations": ["New York"], "source": "openai" }\n\n'
        "Input text:\n\"\"\"\n" + text + "\n\"\"\"\n"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    try:
        content = resp.choices[0].message["content"]
    except Exception:
        content = str(resp)

    # 1) direct parse
    blob = extract_json_blob(content)
    if blob:
        obj, err = _safe_json_load(blob)
        if obj is not None:
            obj.setdefault("source", "openai")
            obj.setdefault("extraction_path", "openai-direct")
            return obj

    # 2) cleaned/unescape parse
    cleaned = _clean_escaped_json(content)
    blob2 = extract_json_blob(cleaned)
    if blob2:
        obj2, err2 = _safe_json_load(blob2)
        if obj2 is not None:
            obj2.setdefault("source", "openai")
            obj2.setdefault("extraction_path", "openai-cleaned-unescape")
            return obj2

    # 3) reformat using model
    reformatted = _reformat_openai_output(content, model=model)
    if isinstance(reformatted, dict) and not reformatted.get("error"):
        reformatted.setdefault("source", "openai")
        reformatted.setdefault("extraction_path", "openai-reformatted")
        return reformatted

    # 4) give up on parsing here
    return {
        "error": "openai-parsing-failed",
        "error_detail": "tried direct, cleaned, and reformat attempts without valid JSON",
        "raw": content,
        "cleaned": cleaned,
        "source": "openai",
        "extraction_path": "openai-parsing-failed"
    }

def merge_openai_spacy(openai_result: Dict, text: str) -> Dict:
    """
    Merge OpenAI results with spaCy fallback results:
    - prefer OpenAI items, but add any missing person/date/location found by spaCy.
    - mark extraction_path to reflect merge.
    """
    merged = {
        "persons": list(openai_result.get("persons", [])) if isinstance(openai_result, dict) else [],
        "dates": list(openai_result.get("dates", [])) if isinstance(openai_result, dict) else [],
        "locations": list(openai_result.get("locations", [])) if isinstance(openai_result, dict) else [],
        "source": openai_result.get("source", "openai"),
        "extraction_path": openai_result.get("extraction_path", "openai")
    }

    # get spaCy/regex candidates
    try:
        spacy_res = extract_with_spacy_and_regex(text)
    except Exception:
        spacy_res = {"persons": [], "dates": [], "locations": []}

    # helper to add missing keeping order
    def add_missing(target_list, candidates):
        seen = set(target_list)
        for c in candidates:
            if c and c not in seen:
                seen.add(c)
                target_list.append(c)

    add_missing(merged["persons"], spacy_res.get("persons", []))
    add_missing(merged["dates"], spacy_res.get("dates", []))
    add_missing(merged["locations"], spacy_res.get("locations", []))

    # if merge added something from spaCy, update extraction_path
    if spacy_res.get("persons") or spacy_res.get("dates") or spacy_res.get("locations"):
        merged["extraction_path"] = merged.get("extraction_path", "") + "+merged-spacy"

    return merged

def process_text_content(contents: str, source_filename: str) -> Dict:
    """
    Primary: call OpenAI. If OpenAI parsing failed or returned empty entities, attempt spaCy fallback.
    Then merge spaCy results into OpenAI output (to catch missing single-token names).
    Save CSV row (including original_text).
    """
    # call OpenAI
    try:
        openai_result = call_openai_extract(contents)
    except Exception as e:
        openai_result = {
            "error": "openai-call-failed",
            "error_detail": str(e),
            "persons": [],
            "dates": [],
            "locations": [],
            "source": "openai",
            "extraction_path": "openai-call-failed"
        }

    # decide whether fallback/merge is needed
    need_merge = False
    if isinstance(openai_result, dict):
        if openai_result.get("error") in ("openai-parsing-failed", "openai-call-failed", "openai-reformat-failed"):
            need_merge = True
        else:
            # if openai found something but may have missed single-token names, we will still merge to be safe
            need_merge = True
    else:
        need_merge = True

    final_result = openai_result
    if need_merge:
        final_result = merge_openai_spacy(openai_result if isinstance(openai_result, dict) else {}, contents)

    # Normalize dates again
    if isinstance(final_result, dict) and "dates" in final_result:
        normalized = []
        for d in final_result.get("dates", []):
            iso = normalize_date_to_iso(d)
            normalized.append(iso if iso else d)
        final_result["dates"] = normalized

    # Ensure arrays and meta fields
    if isinstance(final_result, dict):
        final_result.setdefault("persons", [])
        final_result.setdefault("dates", [])
        final_result.setdefault("locations", [])
        final_result.setdefault("source", final_result.get("source", "openai"))
        final_result.setdefault("extraction_path", final_result.get("extraction_path", "openai"))
    else:
        final_result = {
            "persons": [],
            "dates": [],
            "locations": [],
            "error": "unexpected-result",
            "raw": str(final_result),
            "source": "openai",
            "extraction_path": "unexpected"
        }

    # Save to CSV including original_text
    persons = final_result.get("persons", [])
    dates = final_result.get("dates", [])
    locations = final_result.get("locations", [])
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow({
            "source_filename": source_filename,
            "persons": ";".join(persons) if persons else "",
            "dates": ";".join(dates) if dates else "",
            "locations": ";".join(locations) if locations else "",
            "raw_json": json.dumps(final_result, ensure_ascii=False),
            "original_text": contents.replace("\n", " ").strip()[:10000]  # limit length to 10k chars
        })

    return final_result

# ---------------- endpoints ----------------
@app.get("/")
async def root():
    return {"status": "ok", "endpoints": ["/process (POST)", "/process-batch (POST)", "/process-zip (POST)", "/csv"]}

from fastapi import Header

@app.post("/process")
async def process(
    request: Request,
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    x_api_key: Optional[str] = Header(None)   # 👈 add this
):
    # --- API Key check ---
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    # ----------------------

    # support JSON body
    if not text and not file:
        try:
            body = await request.json()
            if isinstance(body, dict):
                text = body.get("text") or body.get("content")
        except Exception:
            pass

    if file:
        raw = await file.read()
        try:
            contents = raw.decode("utf-8", errors="ignore")
        except Exception:
            contents = str(raw)
        source_filename = file.filename or "uploaded_file"
    elif text:
        contents = text
        source_filename = "inline_text"
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide 'text' (JSON/form) or upload a text file named 'file'."
        )

    result = process_text_content(contents, source_filename)
    return JSONResponse(content=result)


@app.post("/process-batch")
async def process_batch(folder: Optional[str] = Form(None)):
    if not folder:
        raise HTTPException(status_code=400, detail="Provide 'folder' form field (name under ./ingest).")
    if ".." in folder or folder.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid folder name.")
    folder_path = os.path.join(INGEST_BASE, folder)
    if not os.path.isdir(folder_path):
        raise HTTPException(status_code=404, detail=f"Folder {folder_path} not found.")
    results = []
    for fname in sorted(os.listdir(folder_path)):
        if not fname.lower().endswith(".txt"):
            continue
        full = os.path.join(folder_path, fname)
        with open(full, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        res = process_text_content(txt, f"{folder}/{fname}")
        results.append({"file": fname, "result": res})
    return JSONResponse(content={"processed": len(results), "results": results})

@app.post("/process-zip")
async def process_zip(zipfile_upload: UploadFile = File(...)):
    if not zipfile_upload:
        raise HTTPException(status_code=400, detail="Upload a zip file as 'zipfile_upload'.")
    data = await zipfile_upload.read()
    results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for zi in z.infolist():
                    if zi.is_dir():
                        continue
                    name = zi.filename
                    if name.startswith("/") or ".." in name:
                        continue
                    if not name.lower().endswith(".txt"):
                        continue
                    with z.open(zi) as f:
                        raw = f.read()
                        try:
                            txt = raw.decode("utf-8", errors="ignore")
                        except Exception:
                            txt = str(raw)
                    res = process_text_content(txt, f"zip:{name}")
                    results.append({"file": name, "result": res})
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid ZIP.")
    return JSONResponse(content={"processed": len(results), "results": results})

@app.get("/csv")
async def get_csv_preview(limit: int = 50):
    if not os.path.exists(CSV_FILE):
        return JSONResponse(content={"rows": []})
    rows = []
    with open(CSV_FILE, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        for row in reader[-limit:]:
            rows.append(row)
    return JSONResponse(content={"rows": rows})
