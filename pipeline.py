#!/usr/bin/env python3
import os, sys, json, time, signal, random, logging, argparse
from typing import Dict, Any, List, Optional
import boto3
from botocore.exceptions import ClientError, BotoCoreError, EndpointConnectionError
from botocore.config import Config

# ================== LOGGING ==================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}'
)
log = logging.getLogger("ocr-agent")

# ============ CONFIG ============
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")  # prefer IAM role; optional
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")  # prefer IAM role; optional

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Route labels
ROUTE_LABELS = ["bank_statement", "identity", "property", "entity", "loan", "unknown"]

# ============ SAFE SHUTDOWN ============
_SHOULD_STOP = False
def _install_signal_handlers():
    def _handle(sig, frame):
        global _SHOULD_STOP
        _SHOULD_STOP = True
        log.warning(f"Received signal {sig}; preparing to stop gracefully.")
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _handle)
        except Exception:
            pass
_install_signal_handlers()

# ============ OPENAI HELPERS ============
def _strip_json_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl+1:]
        if s.endswith("```"):
            s = s[:-3].strip()
    return s

def _chat_json(model: str, system_text: str, user_payload: dict) -> dict:
    """
    Robust JSON-only helper with structured errors.
    Returns dict. On failure, returns {"_error": "..."}.
    """
    system_msg = (
        system_text
        + "\n\nReturn a single JSON object only. Do not include any extra text."
    )
    user_msg = (
        "You MUST return a single JSON object only (JSON). No prose, no code fences.\n\n"
        "Payload follows as JSON:\n"
        + json.dumps(user_payload, ensure_ascii=False)
    )

    if not OPENAI_API_KEY:
        return {"_error": "OPENAI_API_KEY not set"}

    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=OPENAI_API_KEY)
        try:
            resp = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                temperature=0,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )
            content = resp.choices[0].message.content
            return json.loads(content)
        except Exception as e1:
            log.warning(f"[LLM] strict json_format failed; retrying without response_format: {e1}")
            try:
                resp = client.chat.completions.create(
                    model=model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": system_msg + "\n(You must still return JSON.)"},
                        {"role": "user", "content": user_msg},
                    ],
                )
                raw = resp.choices[0].message.content
                raw_clean = _strip_json_code_fences(raw)
                return json.loads(raw_clean)
            except Exception as e2:
                fname = os.path.join(OUTPUT_DIR, f"llm_error_{int(time.time())}.txt")
                try:
                    with open(fname, "w", encoding="utf-8") as f:
                        f.write(str(e2))
                except Exception:
                    pass
                return {"_error": f"llm_parse_failed: {e2.__class__.__name__}"}
    except Exception as e3:
        try:
            import openai  # type: ignore
            openai.api_key = OPENAI_API_KEY
            try:
                resp = openai.ChatCompletion.create(
                    model=model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                )
                raw = resp["choices"][0]["message"]["content"]
                raw_clean = _strip_json_code_fences(raw)
                return json.loads(raw_clean)
            except Exception as e4:
                return {"_error": f"legacy_llm_parse_failed: {e4.__class__.__name__}"}
        except Exception as e5:
            return {"_error": f"llm_sdk_import_failed: {e5.__class__.__name__}"}

def remove_raw_text_fields(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: remove_raw_text_fields(v) for k, v in obj.items() if k != "raw_text"}
    if isinstance(obj, list):
        return [remove_raw_text_fields(x) for x in obj]
    return obj

# ============ PROMPTS ============
PROMPTS_BY_TYPE: Dict[str, str] = {
    "bank_statement": (
        "You are a financial document parser for BANK STATEMENTS. "
        "Input JSON contains arrays: 'lines', 'cells', and 'kvs'. "
        "Return a single JSON object with anything present (do not limit to a fixed field list). "
        "Prefer structured keys where obvious (e.g., statement_period, balances, transactions). "
        "Additionally, set a top-level field 'document_name' to the exact document title/name as mentioned in the document itself when present; otherwise omit. "
        "Include a 'key_values' array when you see labeled pairs.\n\n"
        "Key–Value guarantees (MUST follow in the output):\n"
        "1) Never leave a value empty. Every key must have a non-empty value.\n"
        "2) If a labeled line contains a colon (e.g., 'ISS: 2020-01-01'), split at the first colon: key='ISS', value='2020-01-01'.\n"
        "3) If a line is an enumerated clause like '(iv) A judicial decree is entered ...', emit key='clause_(iv)' and value='A judicial decree is entered ...'.\n"
        "4) If no explicit value is present and you cannot split or infer, set value equal to the visible key text (mirror the key) so it is not empty.\n"
        "Rules: Preserve text exactly; do not normalize; omit fields not present. Output a single JSON object only."
    ),
    "identity": (
        "You parse identity documents (driver license, state ID, passport, etc). "
        "Input JSON has 'lines', 'cells', and 'kvs'. Extract ALL information present in the document. "
        "Return ONE JSON object in flat key:value form (e.g., 'DOB','EYES','HGT','WGT','SEX','CLASS','ISS','EXP','ORGAN DONOR', etc).\n\n"
        "Value guarantees (MUST follow in the output):\n"
        "1) Never leave a value empty. If a label appears with 'Label: value', use everything after the first colon as the value.\n"
        "2) If a field is indicated only by a symbol (e.g., a heart icon for Organ Donor), use the symbol itself as the value (e.g., '❤️').\n"
        "3) If you cannot find a separate value, set the value equal to the visible field text so that it is not empty.\n"
        "4) For list-style clauses like '(iv) …' that belong to the document content rather than a standard field, add them as 'clause_(iv)': '…'.\n"
        "Rules:\n"
        "- Preserve text exactly as shown (units, casing, punctuation, emojis).\n"
        "- Do not wrap fields as objects; use plain JSON key:value pairs.\n"
        "- Do not invent fields; only include what is clearly present.\n"
        "- Additionally, set a top-level field 'document_name' to the exact document title/name as mentioned in the document itself when present; otherwise omit.\n"
        "- Output JSON only, no prose."
    ),
    "property": (
        "You are an exhaustive parser for PROPERTY-related documents (appraisals, deeds, plats, surveys, covenants, tax records). "
        "Input has 'lines', 'cells', and 'kvs'. Extract EVERYTHING present. Return ONE JSON object only.\n\n"
        "Sections (include when present):\n"
        "  title_block: address, parcel/APN, legal description, borrower/owner/lender, zoning, tax info, subdivision, district/county/state, dates, registry numbers.\n"
        "  valuation: approaches (cost/sales/income), opinion of value, effective date, exposure/marketing time, reconciliations.\n"
        "  site: lot size, utilities, zoning compliance, easements, hazards, topography, influences.\n"
        "  improvements: year built, style, condition (C-ratings), renovations, construction details (foundation, roof, HVAC, windows, floors), amenities (garages, decks, fireplaces).\n"
        "  sales_history: full chain with dates, prices, document types, grantors/grantees, book/page.\n"
        "  comparables: reconstruct comparable tables into arrays with adjustments, net/gross, distances, remarks.\n"
        "  key_values: all labeled pairs as {key, value}.\n"
        "  approvals: signers, roles, license numbers, expirations, certifications, supervisory details.\n"
        "  maps_legends: captions, scales, legends, directional notes.\n"
        "  notes: disclaimers, limiting conditions, free text not captured elsewhere.\n\n"
        "Key–Value guarantees (MUST follow in the output):\n"
        "1) Every {key, value} pair must have a non-empty value.\n"
        "2) If the labeled text includes a colon, split at the first colon into key and value.\n"
        "3) For enumerated clauses '(i)…', '(iv)…', etc., use key='clause_(i)' / 'clause_(iv)' and value='…'.\n"
        "4) If neither split nor inference is possible, copy the visible key text into value (mirror) so it is not empty.\n"
        "Rules: Preserve text EXACTLY; reconstruct tables; include checkboxes/symbols as-is; no prose. "
        "Additionally, set a top-level field 'document_name' to the exact document title/name as mentioned in the document itself when present; otherwise omit."
    ),
    "entity": (
        "You are an exhaustive parser for ENTITY/BUSINESS documents (formation, amendments, certificates, operating agreements, annual reports). "
        "Input has 'lines', 'cells', and 'kvs'. Extract EVERYTHING. Return ONE JSON object only.\n\n"
        "Sections (include when present):\n"
        "  header: document titles, form names/codes, jurisdiction, filing office.\n"
        "  entity_profile: legal name(s), prior names/DBAs, entity type/class, jurisdiction, formation date, duration.\n"
        "  identifiers: EIN, state ID, SOS#, file#, control#, NAICS, DUNS.\n"
        "  registered_agent: name, ID, addresses, consent statements.\n"
        "  addresses: principal office, mailing, records office.\n"
        "  management: organizers, members/managers, directors, officers (names, roles, addresses, terms).\n"
        "  ownership_capital: shares/units/classes, par value, authorized/issued, ownership table.\n"
        "  purpose_powers: stated purpose, limitations, provisions.\n"
        "  compliance: annual reports, franchise tax, effective dates, delayed effectiveness.\n"
        "  approvals: signatures, seals, notary blocks, certifications, filing acknowledgments, dates/times.\n"
        "  key_values: every labeled pair as {key, value}.\n"
        "  tables: any tables reconstructed from 'cells'.\n"
        "  notes: free text not captured elsewhere.\n\n"
        "Key–Value guarantees (MUST follow in the output):\n"
        "1) No empty values. If a label uses a colon, split and use the right side as value.\n"
        "2) Enumerated clauses like '(iv) …' must be emitted as key='clause_(iv)', value='…'.\n"
        "3) If you truly cannot find a separate value, mirror the key text into the value so it is not empty.\n"
        "Rules: Preserve text exactly; reconstruct tables; include checkboxes/symbols; no prose. "
        "Additionally, set a top-level field 'document_name' to the exact document title/name as mentioned in the document itself when present; otherwise omit."
    ),
    "loan": (
        "You are an exhaustive parser for LOAN documents (notes, disclosures, deeds of trust, LE/CD, riders). "
        "Input has 'lines', 'cells', and 'kvs'. Extract EVERYTHING. Return ONE JSON object only.\n\n"
        "Sections (include when present):\n"
        "  parties: borrower(s), lender, trustee, servicer, MERS, guarantors (names, addresses).\n"
        "  loan_terms: principal, interest rate, APR/APY, rate type, index/margin, caps, schedule, maturity, amortization, prepayment, late fees, escrow, balloon, ARM disclosures.\n"
        "  collateral: property address/legal, lien position, riders/addenda.\n"
        "  fees_costs: itemized fees, finance charges, points, credits (reconstruct tables).\n"
        "  disclosures: TILA/RESPA, right to cancel, servicing transfer, privacy, HMDA, ECOA.\n"
        "  compliance_numbers: loan #, application #, NMLS IDs, case numbers.\n"
        "  signatures_notary: signature lines, notary acknowledgments, seals, dates/times.\n"
        "  key_values: every labeled pair as {key, value}.\n"
        "  tables: payment schedules, fee tables, escrow analyses.\n"
        "  notes: any free text not captured elsewhere.\n\n"
        "Key–Value guarantees (MUST follow in the output):\n"
        "1) Every {key, value} pair must have a non-empty value.\n"
        "2) If the key text includes a colon, split once at the first colon.\n"
        "3) For enumerated clauses '(i)…/(iv)…', emit key='clause_(i)' etc., with the remaining text as value.\n"
        "4) If there is still no explicit value, mirror the key text into the value.\n"
        "Rules: Preserve text exactly; reconstruct tables; include checkboxes/symbols; no prose. "
        "Additionally, set a top-level field 'document_name' to the exact document title/name as mentioned in the document itself when present; otherwise omit."
    ),
    "unknown": (
        "You are a cautious yet exhaustive parser for UNKNOWN document types. "
        "Input has 'lines', 'cells', and 'kvs'. Extract EVERYTHING visible without guessing meaning. Return ONE JSON object only.\n\n"
        "Output shape:\n"
        "  key_values: array of {key, value} for any labeled pairs you can see.\n"
        "  free_text: ordered array of textual lines exactly as shown.\n"
        "  tables: reconstructed from 'cells' (array of row objects) when present.\n"
        "  checkmarks: array of {label, status} for selection elements with 'SELECTED' or 'NOT_SELECTED'.\n"
        "  notes: content that is ambiguous or uncategorized.\n\n"
        "Key–Value guarantees (MUST follow in the output):\n"
        "1) No empty values in any {key, value} object.\n"
        "2) If a labeled line contains a colon, split once at the first colon to form value.\n"
        "3) If an enumerated clause like '(iv) …' appears, emit key='clause_(iv)' and value='…'.\n"
        "4) If none of the above applies and you have only a lone label, mirror the label text into value.\n"
        "Additionally, set a top-level field 'document_name' to the exact document title/name as mentioned in the document itself when present; otherwise omit. "
        "Preserve text exactly; do not normalize; no prose."
    ),
}

# ============ AWS CLIENTS ============
_TEXTRACT_CLIENT = None
_S3_CLIENT = None
_BOTO_CONFIG = Config(
    retries={"max_attempts": 5, "mode": "standard"},
    connect_timeout=10,
    read_timeout=60,
)

def _tx():
    global _TEXTRACT_CLIENT
    if _TEXTRACT_CLIENT is None:
        kwargs = {"region_name": AWS_REGION, "config": _BOTO_CONFIG}
        if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
            kwargs["aws_access_key_id"] = AWS_ACCESS_KEY_ID
            kwargs["aws_secret_access_key"] = AWS_SECRET_ACCESS_KEY
        try:
            _TEXTRACT_CLIENT = boto3.client("textract", **kwargs)
        except (BotoCoreError, ClientError) as e:
            raise RuntimeError(f"Failed to initialize Textract client: {e}")
    return _TEXTRACT_CLIENT

def _s3():
    global _S3_CLIENT
    if _S3_CLIENT is None:
        kwargs = {"region_name": AWS_REGION, "config": _BOTO_CONFIG}
        if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
            kwargs["aws_access_key_id"] = AWS_ACCESS_KEY_ID
            kwargs["aws_secret_access_key"] = AWS_SECRET_ACCESS_KEY
        try:
            _S3_CLIENT = boto3.client("s3", **kwargs)
        except (BotoCoreError, ClientError) as e:
            raise RuntimeError(f"Failed to initialize S3 client: {e}")
    return _S3_CLIENT

def _aws_sanity_check():
    """Verify AWS region/creds are usable; log caller identity."""
    try:
        sts = boto3.client(
            "sts",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID or None,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY or None,
            config=_BOTO_CONFIG,
        )
        ident = sts.get_caller_identity()
        log.info(f"AWS caller identity ok: {ident.get('Account')} / {ident.get('Arn')}")
    except Exception as e:
        raise RuntimeError(f"AWS credentials/region check failed: {e}")

def _s3_head(bucket: str, key: str):
    """Ensure the S3 object exists and is readable."""
    try:
        s3 = _s3()
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code == 404:
            raise FileNotFoundError(f"S3 object not found: s3://{bucket}/{key}")
        raise RuntimeError(f"S3 head_object failed: {e}")

# ============ TEXTRACT ============
def _sleep_backoff(attempt: int, base: float = 2.0, max_sleep: float = 10.0):
    sleep = min(max_sleep, base ** attempt) + random.uniform(0, 0.5)
    time.sleep(sleep)

def run_textract_async_s3(bucket: str, key: str, max_wait_seconds: int = 600) -> Dict[str, Any]:
    _s3_head(bucket, key)
    client = _tx()
    try:
        response = client.start_document_analysis(
            DocumentLocation={"S3Object": {"Bucket": bucket, "Name": key}},
            FeatureTypes=["TABLES", "FORMS"],
        )
        job_id = response["JobId"]
        log.info(f"[Textract] Started job {job_id}")
    except (BotoCoreError, ClientError, EndpointConnectionError) as e:
        raise RuntimeError(f"Textract start failed: {e}")

    start_time = time.time()
    attempts = 0
    while True:
        if _SHOULD_STOP:
            raise RuntimeError(f"Aborted by signal while polling Textract job {job_id}")
        try:
            status = client.get_document_analysis(JobId=job_id)
        except (BotoCoreError, ClientError, EndpointConnectionError) as e:
            attempts += 1
            if time.time() - start_time > max_wait_seconds:
                raise TimeoutError(f"Textract job {job_id} polling timed out after {max_wait_seconds}s; last error: {e}")
            log.warning(f"[Textract] Polling error (attempt {attempts}): {e}; backing off…")
            _sleep_backoff(attempts)
            continue

        job_status = status.get("JobStatus")
        if job_status in ["SUCCEEDED", "FAILED", "PARTIAL_SUCCESS"]:
            break
        if time.time() - start_time > max_wait_seconds:
            raise TimeoutError(f"Textract job {job_id} timed out after {max_wait_seconds}s (elapsed={int(time.time()-start_time)}s)")
        time.sleep(3)

    if job_status == "FAILED":
        raise RuntimeError(f"Textract job failed: {job_id}")
    if job_status == "PARTIAL_SUCCESS":
        log.warning(f"[Textract] Job {job_id} returned PARTIAL_SUCCESS")

    blocks: List[Dict[str, Any]] = []
    next_token = None
    pages_total = status.get("DocumentMetadata", {}).get("Pages")
    while True:
        try:
            if next_token:
                status = client.get_document_analysis(JobId=job_id, NextToken=next_token)
            else:
                status = client.get_document_analysis(JobId=job_id)
        except (BotoCoreError, ClientError, EndpointConnectionError) as e:
            raise RuntimeError(f"Textract pagination failed for job {job_id}: {e}")
        blocks.extend(status.get("Blocks", []))
        next_token = status.get("NextToken")
        if not next_token:
            break

    return {
        "engine_meta": {
            "mode": "textract:start_document_analysis",
            "pages": pages_total,
            "job_id": job_id,
            "elapsed_sec": int(time.time() - start_time),
        },
        "blocks": blocks,
    }

def run_analyze_id_s3(bucket: str, key: str) -> Dict[str, Any]:
    image_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".heic", ".webp")
    if not key.lower().endswith(image_exts):
        raise RuntimeError("AnalyzeID is only attempted for image files (png/jpg/tiff/etc).")
    _s3_head(bucket, key)
    client = _tx()
    try:
        resp = client.analyze_id(DocumentPages=[{"S3Object": {"Bucket": bucket, "Name": key}}])
        return resp
    except (BotoCoreError, ClientError, EndpointConnectionError) as e:
        raise RuntimeError(f"AnalyzeID failed: {e}")

# ============ BLOCK UTILS ============
def _group_blocks_by_page(blocks: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    pages: Dict[int, List[Dict[str, Any]]] = {}
    for b in blocks:
        page = b.get("Page", 1)
        pages.setdefault(page, []).append(b)
    return pages

def _resolve_kv_pairs_from_page_blocks(page_blocks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    by_id = {b.get("Id"): b for b in page_blocks if b.get("Id")}
    kvs: List[Dict[str, str]] = []

    for b in page_blocks:
        if b.get("BlockType") != "KEY_VALUE_SET":
            continue
        entity = b.get("EntityTypes", []) or []
        if "KEY" not in entity:
            continue

        key_text_parts: List[str] = []
        value_text_parts: List[str] = []

        for rel in b.get("Relationships", []) or []:
            rtype = rel.get("Type")
            if rtype == "CHILD":
                for cid in rel.get("Ids", []) or []:
                    cb = by_id.get(cid, {})
                    t = cb.get("Text")
                    if t:
                        key_text_parts.append(t)

            elif rtype == "VALUE":
                for vid in rel.get("Ids", []) or []:
                    vb = by_id.get(vid, {})
                    for vrel in vb.get("Relationships", []) or []:
                        if vrel.get("Type") == "CHILD":
                            for vcid in vrel.get("Ids", []) or []:
                                vcb = by_id.get(vcid, {})
                                if vcb.get("BlockType") == "SELECTION_ELEMENT":
                                    status = vcb.get("SelectionStatus")
                                    if status:
                                        value_text_parts.append(f"[CHECKBOX:{status}]")
                                else:
                                    vt = vcb.get("Text")
                                    if vt:
                                        value_text_parts.append(vt)

        k = " ".join(key_text_parts).strip()
        v = " ".join(value_text_parts).strip()
        if k or v:
            kvs.append({"key": k, "value": v})

    return kvs

def _cells_from_page_blocks(page_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id = {blk.get("Id"): blk for blk in page_blocks if blk.get("Id")}
    cells: List[Dict[str, Any]] = []
    for b in page_blocks:
        if b.get("BlockType") != "CELL":
            continue
        words: List[str] = []
        lines: List[str] = []
        checks: List[str] = []
        for rel in b.get("Relationships", []) or []:
            if rel.get("Type") != "CHILD":
                continue
            for cid in rel.get("Ids", []) or []:
                cb = by_id.get(cid, {})
                bt = cb.get("BlockType")
                if bt == "WORD":
                    t = cb.get("Text")
                    if t:
                        words.append(t)
                elif bt == "LINE":
                    t = cb.get("Text")
                    if t:
                        lines.append(t)
                elif bt == "SELECTION_ELEMENT":
                    status = cb.get("SelectionStatus")
                    if status:
                        checks.append(f"[CHECKBOX:{status}]")
        cell_text = " ".join(words) if words else " ".join(lines)
        if checks:
            cell_text = (cell_text + " " + " ".join(checks)).strip()
        cells.append({
            "row": b.get("RowIndex"),
            "col": b.get("ColumnIndex"),
            "text": cell_text.strip(),
        })
    return cells

def _lines_words_from_page_blocks(page_blocks: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    lines, words = [], []
    for b in page_blocks:
        bt = b.get("BlockType")
        if bt == "LINE":
            t = b.get("Text", "")
            if t:
                lines.append(t)
        elif bt == "WORD":
            t = b.get("Text", "")
            if t:
                words.append(t)
    return {"lines": lines, "words": words}

# ============ ROUTER ============
def llm_route_from_ocr_page(page_view: Dict[str, Any]) -> str:
    lines = page_view.get("lines", [])
    cells = page_view.get("cells", [])
    payload = {
        "head_lines": lines[:120],
        "tail_lines": lines[-40:],
        "cells_preview": [
            {"row": c.get("row"), "col": c.get("col"), "text": (c.get("text") or "")[:120]}
            for c in cells[:120]
        ],
    }
    system = (
        "You are a cautious document-type classifier for business and identity documents. "
        f"Choose exactly one label from {ROUTE_LABELS}. "
        'Return a single JSON object exactly in the form {"doc_type":"<label>"}. '
        "Rules:\n"
        "- If not highly confident, return 'unknown'. Never guess.\n"
        "- Base your decision only on the provided snippets.\n"
        "- Only output a JSON object; no extra text."
    )
    out = _chat_json(OPENAI_MODEL, system, payload) or {}
    label = out.get("doc_type", "unknown")
    if "_error" in out:
        log.warning(f"[Router] LLM error: {out['_error']}; defaulting to 'unknown'")
        return "unknown"
    return label if label in ROUTE_LABELS else "unknown"

def route_document_type_from_ocr(simplified: Dict[str, Any]) -> str:
    if not simplified.get("pages"):
        return "unknown"
    page_keys = sorted(simplified["pages"].keys(), key=lambda x: int(x))[:3]
    votes: Dict[str, int] = {}
    for pk in page_keys:
        label = llm_route_from_ocr_page(simplified["pages"][pk])
        votes[label] = votes.get(label, 0) + 1
    label = max(votes, key=votes.get) if votes else "unknown"
    log.info(f"[Router] votes={votes} -> doc_type={label}")
    return label

# ============ LLM EXTRACTION ============
def llm_extract_page(doc_type: str, page_data: Dict[str, Any]) -> Dict[str, Any]:
    llm_input = {
        "lines": page_data.get("lines", []),
        "cells": page_data.get("cells", []),
        "kvs":   page_data.get("kvs", []),
    }
    system = PROMPTS_BY_TYPE.get(doc_type, PROMPTS_BY_TYPE["unknown"])
    out = _chat_json(OPENAI_MODEL, system, llm_input) or {}
    if "_error" in out:
        fname = os.path.join(OUTPUT_DIR, f"llm_extract_error_{doc_type}_{int(time.time())}.json")
        try:
            with open(fname, "w", encoding="utf-8") as f:
                json.dump({"input": llm_input, "error": out["_error"]}, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
        return {"_error": out["_error"]}
    return remove_raw_text_fields(out)

# ============ LLM IMAGE ============
def _classify_via_image(model: str, image_url: str) -> str:
    system = (
        "You are a cautious document-type classifier for business and identity documents. "
        f"Choose exactly one label from {ROUTE_LABELS}. "
        'Return a single JSON object exactly in the form {"doc_type":"<label>"}. '
        "Rules: If not highly confident, return 'unknown'."
    )
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "text", "text": "Classify this document. Return {\"doc_type\":\"<label>\"}."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]},
            ],
        )
        content = resp.choices[0].message.content
        out = json.loads(content)
        label = out.get("doc_type", "unknown")
        return label if label in ROUTE_LABELS else "unknown"
    except Exception as e:
        log.error(f"[LLM-IMG] classification failed: {e}")
        return "unknown"

def _extract_via_image(model: str, doc_type: str, image_url: str) -> Dict[str, Any]:
    system = (
        PROMPTS_BY_TYPE.get(doc_type, PROMPTS_BY_TYPE["unknown"]) +
        "\nReturn ONE JSON object only."
    )
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "text", "text": "Extract all structured data from this document image per the rules."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]},
            ],
        )
        content = resp.choices[0].message.content
        try:
            return remove_raw_text_fields(json.loads(content))
        except Exception as e:
            log.error(f"[LLM-IMG] parse failed: {e}")
            return {"_error": f"llm_image_parse_failed: {e.__class__.__name__}"}
    except Exception as e:
        log.error(f"[LLM-IMG] extraction failed: {e}")
        return {"_error": f"llm_image_failed: {e.__class__.__name__}"}

# ============ ANALYZEID CONVERSION ============
def analyze_id_to_kvs(resp: Dict[str, Any]) -> List[Dict[str, str]]:
    kvs: List[Dict[str, str]] = []
    try:
        docs = resp.get("IdentityDocuments", [])
        for doc in docs:
            dtype = doc.get("DocumentType")
            if dtype:
                kvs.append({"key": "DocumentType", "value": str(dtype)})
            for f in doc.get("Fields", []):
                k = (f.get("Type", {}) or {}).get("Text", "")
                v = (f.get("ValueDetection", {}) or {}).get("Text", "")
                if k or v:
                    kvs.append({"key": k, "value": v})
    except Exception as e:
        log.warning(f"[AnalyzeID] convert failed: {e}")
    return kvs

# ============ MAIN PIPELINE ============
def run_pipeline(bucket: str, key: str, mode: str = "ocr+llm") -> Dict[str, Any]:
    if not AWS_REGION:
        log.warning("[Warn] AWS_REGION not set; using boto3 defaults.")
    if not OPENAI_API_KEY:
        log.warning("[Warn] OPENAI_API_KEY not set. LLM steps will return error objects.")

    _aws_sanity_check()
    _s3_head(bucket, key)

    simplified: Dict[str, Any] = {"pages": {}}
    name_no_ext = os.path.splitext(os.path.basename(key))[0]
    start_total = time.time()

    try:
        if mode == "ocr+llm":
            raw = run_textract_async_s3(bucket, key)
            blocks = raw.get("blocks", [])
            pages_full = _group_blocks_by_page(blocks)

            simplified = {"pages": {}}
            for page_num, page_blocks in pages_full.items():
                lw = _lines_words_from_page_blocks(page_blocks)
                cells = _cells_from_page_blocks(page_blocks)
                kvs = _resolve_kv_pairs_from_page_blocks(page_blocks)

                simplified["pages"][page_num] = {
                    "lines": lw["lines"],
                    "words": lw["words"],
                    "cells": cells,
                    "kvs": kvs,
                }

            doc_type = route_document_type_from_ocr(simplified)
            log.info(f"[Router] Document type: {doc_type}")

            if doc_type == "identity":
                try:
                    aid = run_analyze_id_s3(bucket, key)
                    aid_kvs = analyze_id_to_kvs(aid)
                    first_page_key = sorted(simplified["pages"].keys(), key=lambda x: int(x))[0]
                    simplified["pages"][first_page_key].setdefault("kvs", [])
                    simplified["pages"][first_page_key]["kvs"].extend(aid_kvs)

                    aid_file = os.path.join(OUTPUT_DIR, f"analyzeid_{name_no_ext}.json")
                    with open(aid_file, "w", encoding="utf-8") as f:
                        json.dump(aid, f, indent=2, ensure_ascii=False)
                    log.info(f"=== ANALYZEID saved -> {aid_file} ===")
                except Exception as e:
                    log.warning(f"[AnalyzeID] skipped/failed: {e}")

            image_extracted = {}

        else:  # "llm" mode
            lower_key = key.lower()
            if lower_key.endswith(".pdf"):
                log.info("[Mode=llm] PDF detected -> Textract for text + LLM (hybrid).")
                raw = run_textract_async_s3(bucket, key)
                blocks = raw.get("blocks", [])
                pages_full = _group_blocks_by_page(blocks)
                simplified = {"pages": {}}
                for page_num, page_blocks in pages_full.items():
                    lw = _lines_words_from_page_blocks(page_blocks)
                    cells = _cells_from_page_blocks(page_blocks)
                    kvs = _resolve_kv_pairs_from_page_blocks(page_blocks)
                    simplified["pages"][page_num] = {
                        "lines": lw["lines"],
                        "words": lw["words"],
                        "cells": cells,
                        "kvs": kvs,
                    }
                doc_type = route_document_type_from_ocr(simplified)
                log.info(f"[Router] Document type: {doc_type}")
                image_extracted = {}
            else:
                image_url = None
                try:
                    s3c = _s3()
                    image_url = s3c.generate_presigned_url(
                        ClientMethod="get_object",
                        Params={"Bucket": bucket, "Key": key},
                        ExpiresIn=900,
                    )
                except Exception as e:
                    log.error(f"[Warn] Could not create presigned URL: {e}")

                if image_url and OPENAI_API_KEY:
                    doc_type = _classify_via_image(OPENAI_MODEL, image_url)
                    log.info(f"[Router] Document type (image): {doc_type}")
                    image_extracted = _extract_via_image(OPENAI_MODEL, doc_type, image_url)
                    simplified = {"pages": {1: {"lines": [], "words": [], "cells": [], "kvs": []}}}
                else:
                    doc_type = "unknown"
                    simplified = {"pages": {1: {"lines": [], "words": [], "cells": [], "kvs": []}}}
                    image_extracted = {}

        all_structured: Dict[str, Any] = {"doc_type": doc_type, "_metrics": {}}
        doc_name_candidates: List[str] = []

        for page_num in sorted(simplified["pages"].keys(), key=lambda x: int(x)):
            page_data = simplified["pages"][page_num]
            log.info(f"[LLM] Extracting page {page_num} as '{doc_type}'...")
            if mode == "llm" and page_num == 1 and image_extracted:
                extracted = image_extracted
            else:
                extracted = llm_extract_page(doc_type, page_data)

            if isinstance(extracted, dict):
                dn = extracted.get("document_name")
                if dn:
                    doc_name_candidates.append(dn)
                    extracted.pop("document_name", None)

            all_structured[str(page_num)] = extracted

        if doc_name_candidates:
            all_structured["document_name"] = doc_name_candidates[0]
            if len(set(doc_name_candidates)) > 1:
                all_structured["_document_name_candidates"] = list(dict.fromkeys(doc_name_candidates))

        out_name = f"ocr_llm_structured_{name_no_ext}.json" if mode == "ocr+llm" else f"llm_structured_{name_no_ext}.json"
        struct_file = os.path.join(OUTPUT_DIR, out_name)
        all_structured["_metrics"]["elapsed_total_sec"] = int(time.time() - start_total)
        try:
            with open(struct_file, "w", encoding="utf-8") as f:
                json.dump(all_structured, f, indent=2, ensure_ascii=False)
        except OSError as e:
            raise RuntimeError(f"Failed to write output JSON '{struct_file}': {e}")
        log.info(f"=== FINAL STRUCTURED RESULT saved -> {struct_file} ===")

        return {
            "result_path": struct_file,
            "doc_type": doc_type,
            "structured": all_structured,
            "name_no_ext": name_no_ext,
            "mode": mode,
        }

    except TimeoutError as e:
        log.error(f"[Timeout] {e}")
        raise
    except (RuntimeError, FileNotFoundError) as e:
        log.error(f"[Runtime] {e}")
        raise
    except KeyboardInterrupt:
        log.warning("[Abort] KeyboardInterrupt received; exiting.")
        raise
    except Exception as e:
        log.exception(f"[Unexpected] {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Extract data from documents using OCR+LLM or LLM-only.")
    parser.add_argument("arg1", nargs="?", help="S3 bucket (required for both modes)")
    parser.add_argument("arg2", nargs="?", help="S3 key (required for both modes)")
    parser.add_argument("--mode", choices=["ocr+llm", "llm"], default="ocr+llm", help="Extraction mode")
    args = parser.parse_args()

    if not (args.arg1 and args.arg2):
        exe = os.path.basename(sys.argv[0])
        print(f"Usage: python {exe} <s3-bucket> <s3-key> --mode <ocr+llm|llm>")
        sys.exit(1)

    if not AWS_REGION:
        log.warning("[Warn] AWS_REGION not set; defaulting to boto3's configuration chain.")

    try:
        _ = run_pipeline(args.arg1, args.arg2, args.mode)
        sys.exit(0)
    except TimeoutError:
        sys.exit(2)
    except (RuntimeError, FileNotFoundError):
        sys.exit(3)
    except KeyboardInterrupt:
        sys.exit(130)  # 128+SIGINT
    except Exception:
        sys.exit(4)

if __name__ == "__main__":
    main()
