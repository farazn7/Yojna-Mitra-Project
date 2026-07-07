import os
import fitz  # PyMuPDF
import ollama
from PIL import Image
import json
import re

# ---------------------------------------------------------------------------
# 1. DOCUMENT-SPECIFIC TARGETED EXTRACTION PROMPTS
# ---------------------------------------------------------------------------
EXTRACTION_PROMPTS = {
    "aadhaar": (
        "This is an Indian Aadhaar card. Extract ONLY the CARD HOLDER's personal info as JSON:\n"
        '{"aadhaar_number": "the 12-digit number (XXXX XXXX XXXX)", '
        '"full_name_en": "name in English", '
        '"full_name_hi": "name in Hindi/Devanagari", '
        '"date_of_birth": "DD/MM/YYYY", '
        '"gender": "Male/Female/Other", '
        '"address": "street/locality", '
        '"city": "city or town", '
        '"district": "district", '
        '"state": "state", '
        '"pincode": "6-digit PIN"}\n'
        "Output ONLY valid JSON. If a field is not visible, omit that key.\n"
        "IMPORTANT: Do NOT include any organizational, government header, or website text."
    ),
    "pan_card": (
        "This is an Indian PAN card. Extract ONLY the CARD HOLDER's personal info as JSON:\n"
        '{"pan_number": "the 10-character alphanumeric PAN", '
        '"full_name_en": "name in English", '
        '"father_name": "father name", '
        '"date_of_birth": "DD/MM/YYYY"}\n'
        "Output ONLY valid JSON. If a field is not visible, omit that key."
    ),
    "college_id": (
        "This is a college/university student ID card. Extract ONLY the STUDENT's personal info as JSON:\n"
        '{"full_name_en": "student name only", '
        '"institution": "college/university name", '
        '"course": "degree program", '
        '"roll_number": "enrollment/roll number", '
        '"valid_till": "expiry date if visible", '
        '"blood_group": "blood group if visible", '
        '"date_of_birth": "DD/MM/YYYY if visible"}\n'
        "Output ONLY valid JSON. If a field is not visible, omit that key.\n"
        "CRITICAL: Do NOT include the college's phone number, email, or website. Only include the STUDENT's personal details."
    ),
    "voter_id": (
        "This is an Indian Voter ID (EPIC). Extract ONLY the CARD HOLDER's personal info as JSON:\n"
        '{"voter_id_number": "the EPIC number (3 letters + 7 digits)", '
        '"full_name_en": "name in English", '
        '"father_name": "father/husband name", '
        '"date_of_birth": "DD/MM/YYYY or age", '
        '"gender": "Male/Female", '
        '"address": "street/locality", '
        '"city": "city or town", '
        '"state": "state", '
        '"pincode": "6-digit PIN"}\n'
        "Output ONLY valid JSON. If a field is not visible, omit that key."
    ),
    "income_certificate": (
        "This is an Indian Income Certificate. Extract ONLY these fields as JSON:\n"
        '{"full_name_en": "applicant name", '
        '"annual_income": "total annual income in INR (numbers only)", '
        '"issuing_authority": "authority/tehsildar name", '
        '"certificate_number": "certificate ID/number", '
        '"valid_year": "financial year or validity date"}\n'
        "Output ONLY valid JSON. If a field is not visible, omit that key."
    ),
    "caste_certificate": (
        "This is an Indian Caste/Community Certificate. Extract ONLY these fields as JSON:\n"
        '{"full_name_en": "applicant name", '
        '"caste_category": "caste category (e.g., SC, ST, OBC, General)", '
        '"caste_name": "specific community/caste name", '
        '"certificate_number": "certificate ID/number", '
        '"issuing_authority": "authority name"}\n'
        "Output ONLY valid JSON. If a field is not visible, omit that key."
    ),
    "ration_card": (
        "This is an Indian Ration Card. Extract ONLY these fields as JSON:\n"
        '{"ration_card_number": "ration card number", '
        '"head_of_family": "name of head of family", '
        '"card_type": "type of card (e.g., BPL, APL, Antyodaya, PHH)", '
        '"address": "street/locality", '
        '"city": "city or town", '
        '"state": "state", '
        '"pincode": "6-digit PIN"}\n'
        "Output ONLY valid JSON. If a field is not visible, omit that key."
    ),
    "passport": (
        "This is an Indian Passport. Extract ONLY the HOLDER's personal info as JSON:\n"
        '{"passport_number": "passport number", '
        '"full_name_en": "given name plus surname", '
        '"date_of_birth": "DD/MM/YYYY", '
        '"gender": "M/F/X", '
        '"expiry_date": "DD/MM/YYYY"}\n'
        "Output ONLY valid JSON. If a field is not visible, omit that key."
    ),
    "back_of_card": (
        "This is the BACK/REVERSE side of an Indian ID card. "
        "Extract ONLY the CARD HOLDER's personal details as JSON:\n"
        '{"address": "street/locality", '
        '"city": "city or town", '
        '"district": "district", '
        '"state": "state", '
        '"pincode": "6-digit PIN code", '
        '"phone": "personal phone number if visible", '
        '"date_of_birth": "DD/MM/YYYY if visible", '
        '"blood_group": "blood group if visible", '
        '"valid_till": "expiry/validity date if visible"}\n'
        "Output ONLY valid JSON. If a field is not visible, omit that key.\n"
        "CRITICAL: Do NOT include organizational info like college phone, college email, website URLs, "
        "or any text that belongs to the issuing institution rather than the card holder."
    ),
    "unknown": (
        "Extract ONLY the PERSON's personal details from this document as JSON key-value pairs. "
        "Include only: name, date of birth, address, phone, ID numbers, gender, blood group. "
        "Do NOT include organizational info (emails, websites, office numbers). "
        "Use lowercase_underscore keys. Output ONLY valid JSON."
    )
}

# ---------------------------------------------------------------------------
# 2. STRUCTURAL VALIDATION RULES (REGEX)
# ---------------------------------------------------------------------------
VALIDATORS = {
    "aadhaar_number": {
        "pattern": r'^\d{4}\s?\d{4}\s?\d{4}$',
        "clean": lambda s: re.sub(r'\s', '', str(s)),
        "description": "12 digits (e.g., 1234 5678 9012)"
    },
    "pan_number": {
        "pattern": r'^[A-Z]{5}[0-9]{4}[A-Z]$',
        "clean": lambda s: str(s).upper().strip(),
        "description": "5 letters + 4 digits + 1 letter (e.g., ABCDE1234F)"
    },
    "voter_id_number": {
        "pattern": r'^[A-Z]{3}[0-9]{7}$',
        "clean": lambda s: str(s).upper().strip(),
        "description": "3 letters + 7 digits (e.g., ABC1234567)"
    },
    "pincode": {
        "pattern": r'^[1-9][0-9]{5}$',
        "clean": lambda s: str(s).strip(),
        "description": "6-digit Indian PIN code"
    }
}

def validate_extracted_field(field_name: str, value: any) -> tuple[bool, str]:
    """Returns (is_valid, cleaned_value_or_error).
    
    CRITICAL RULE: Only validate fields that were actually found with a real value.
    If the LLM returned None, empty string, or whitespace for a field, that just
    means the field wasn't visible on this document — skip it, don't fail on it.
    """
    # Skip validation entirely for missing/empty values
    if value is None or str(value).strip() == "":
        return True, None  # Field absent — not an error, just omit from cleaned data
    
    rule = VALIDATORS.get(field_name)
    if not rule:
        return True, value  # No validation rule -> pass through untouched

    cleaned = rule["clean"](value)
    if re.match(rule["pattern"], cleaned):
        return True, cleaned
    else:
        return False, f"❌ Invalid {field_name}: '{value}' doesn't match {rule['description']}"

def validate_extracted_data(data: dict, doc_type: str) -> tuple[bool, list[str], dict]:
    """Validates extracted fields against structural regex rules.
    
    Only validates fields that have actual non-empty values.
    Empty/null fields are silently dropped — they represent 'not visible on this doc'.
    """
    is_valid = True
    errors = []
    cleaned_data = {}

    for key, val in data.items():
        field_valid, result = validate_extracted_field(key, val)
        if field_valid:
            if result is not None:  # Only keep fields with actual values
                cleaned_data[key] = result
        else:
            is_valid = False
            errors.append(result)
            cleaned_data[key] = val  # Keep original even if invalid for inspection

    return is_valid, errors, cleaned_data


# ---------------------------------------------------------------------------
# 3. IMAGE PREPARATION HELPER
# ---------------------------------------------------------------------------
def _get_image_for_vision(file_path: str) -> tuple[str, bool]:
    """Helper: If file is a PDF, render page 0 to an image for VLM. Otherwise return as-is.
    Returns (image_path, is_temporary_render).
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        doc = fitz.open(file_path)
        page = doc[0]
        pix = page.get_pixmap(dpi=200)
        temp_img_path = file_path.replace(".pdf", "_render.jpg")
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img.save(temp_img_path)
        doc.close()
        return temp_img_path, True
    return file_path, False

# ---------------------------------------------------------------------------
# 4. PASS 0: DOCUMENT CLASSIFICATION
# ---------------------------------------------------------------------------
def classify_document(file_path: str) -> str:
    """Pass 0: Ask Gemma 3 Vision to identify what kind of Indian document this is."""
    print(f"[PASS 0: CLASSIFY] Analyzing document identity with Gemma 3...")
    
    img_path, is_temp = _get_image_for_vision(file_path)
    
    try:
        response = ollama.chat(
            model='gemma3:4b',
            messages=[{
                'role': 'user',
                'content': (
                    'Look at this image and identify the type of Indian document. '
                    'Reply with EXACTLY ONE of these labels, nothing else:\n'
                    'aadhaar, pan_card, voter_id, driving_license, '
                    'college_id, income_certificate, caste_certificate, '
                    'ration_card, passport, back_of_card, unknown\n\n'
                    'IMPORTANT: If this image shows the BACK/REVERSE side of any ID card '
                    '(address side, barcode side, instruction side), always reply: back_of_card\n'
                    'Only reply "aadhaar" if you can clearly see the UIDAI logo or the words "Aadhaar" or "आधार".'
                ),
                'images': [img_path]
            }],
            options={'temperature': 0.0}
        )
        label = response['message']['content'].strip().lower()
        
        # Match against known labels
        known_types = list(EXTRACTION_PROMPTS.keys()) + ['back_of_card']
        for known_type in known_types:
            if known_type in label:
                print(f"[PASS 0: CLASSIFY] Document identified as: '{known_type}'")
                return known_type
                
        print(f"[PASS 0: CLASSIFY] Could not match label, defaulting to 'unknown' (Raw: '{label}')")
        return "unknown"
        
    finally:
        if is_temp and os.path.exists(img_path):
            os.remove(img_path)


# ---------------------------------------------------------------------------
# 5. PASS 1: TARGETED EXTRACTION
# ---------------------------------------------------------------------------
def extract_with_vision(file_path: str, doc_type: str = "unknown") -> str:
    """Pass 1: Uses Gemma 3 Vision with targeted document-specific prompts."""
    print(f"[PASS 1: EXTRACT] Running targeted extraction for '{doc_type}' using Gemma 3...")
    
    prompt = EXTRACTION_PROMPTS.get(doc_type, EXTRACTION_PROMPTS["unknown"])
    img_path, is_temp = _get_image_for_vision(file_path)
    
    try:
        response = ollama.chat(
            model='gemma3:4b',
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [img_path]
            }],
            options={'temperature': 0.0}
        )
        return response['message']['content'].strip()
    finally:
        if is_temp and os.path.exists(img_path):
            os.remove(img_path)

def extract_from_pdf(file_path: str, doc_type: str = "unknown") -> str:
    """Tries fast PyMuPDF digital text extraction first. Falls back to Vision if scanned."""
    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        text += page.get_text()
    doc.close()
    
    # If it has significant digital text, return it directly
    if len(text.strip()) > 50:
        print("[PASS 1: EXTRACT] Digital PDF detected. Using fast text layer extraction.")
        return text.strip()
        
    # Otherwise it's a scanned PDF -> use Vision
    print("[PASS 1: EXTRACT] Scanned PDF detected. Switching to AI Vision extraction.")
    return extract_with_vision(file_path, doc_type)

def extract_text_from_document(file_path: str, doc_type: str = "unknown") -> str:
    """Routes to digital PDF extractor or Vision extractor based on extension."""
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.pdf':
            return extract_from_pdf(file_path, doc_type)
        elif ext in ['.jpg', '.jpeg', '.png']:
            return extract_with_vision(file_path, doc_type)
        else:
            return "❌ Unsupported file type for extraction."
    except Exception as e:
        return f"❌ Extraction failed: {str(e)}"

# ---------------------------------------------------------------------------
# 6. PASS 2: STRUCTURE & VALIDATE
# ---------------------------------------------------------------------------
def structure_raw_text_to_json(raw_text: str, doc_type: str = "unknown") -> dict:
    """Pass 2: Uses Llama 3.1 to clean raw extraction into perfect JSON."""
    print(f"[PASS 2: STRUCTURE] Handing raw text to Llama 3.1 for JSON formatting...")
    
    prompt = f"""You are a strict data-extraction API. I am going to give you text extracted from an Indian '{doc_type}' document.

Your job is to output ONLY the CARD HOLDER's personal information as a clean JSON object.

Rules:
1. Output ONLY valid JSON. No markdown, no explanations, no backticks.
2. Convert all keys to lowercase_underscores.
3. If a value is unreadable or missing, do NOT include that key.
4. NEVER include organizational data (college email, college phone, website URLs, office addresses).
   Only include data that belongs to the PERSON whose name is on the card.
5. If you see an address, split it into separate keys:
   - "address" for street/locality/apartment
   - "city" for city/town name
   - "district" for district (if separate from city)
   - "state" for the state name
   - "pincode" for the 6-digit PIN code
   Only include address components that are actually present. Do not guess.
6. Phone numbers: Only include the card holder's personal number, not the organization's number.

Raw Text to process:
{raw_text}"""
    
    try:
        response = ollama.chat(
            model='llama3.1',
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.0}
        )
        
        result_text = response['message']['content'].strip()
        if result_text.startswith("```json"):
            result_text = result_text.replace("```json", "", 1)
        if result_text.startswith("```"):
            result_text = result_text.replace("```", "", 1)
        if result_text.endswith("```"):
            result_text = result_text[:result_text.rfind("```")]
            
        return json.loads(result_text.strip())
    except Exception as e:
        print(f"[ERROR] JSON structuring failed ({e}). Returning raw text wrap.")
        return {"raw_extraction": raw_text}

# ---------------------------------------------------------------------------
# 7. MAIN ORCHESTRATOR: THE 3-PASS PIPELINE
# ---------------------------------------------------------------------------
def analyze_document(file_path: str, expected_doc_type: str = None) -> dict:
    """The Complete 3-Pass Document Intelligence Pipeline:
    
    Pass 0 (Classify): Identifies document type (or verifies against expected_doc_type).
    Pass 1 (Targeted Extract): Extracts text using document-specific VLM prompts.
    Pass 2 (Structure & Validate): Cleans into JSON via Llama 3.1 and validates ID numbers via Regex.
    
    Returns:
    {
        "success": bool,
        "doc_type": str,
        "extracted_data": dict,
        "raw_text": str,
        "is_valid": bool,
        "validation_errors": list[str]
    }
    """
    try:
        # Pass 0: Classify document
        detected_type = classify_document(file_path)
        
        # If LangGraph asked for a specific doc type, verify if it matches
        doc_type = detected_type
        if expected_doc_type and expected_doc_type != "unknown":
            if detected_type != "unknown" and detected_type != expected_doc_type:
                print(f"[WARNING] Expected '{expected_doc_type}' but detected '{detected_type}'. Proceeding with detected type.")
            elif detected_type == "unknown":
                doc_type = expected_doc_type  # Trust expected type if VLM was unsure
                
        # Pass 1: Targeted Extract
        raw_text = extract_text_from_document(file_path, doc_type)
        if raw_text.startswith("❌"):
            return {
                "success": False, "doc_type": doc_type, "extracted_data": {},
                "raw_text": raw_text, "is_valid": False, "validation_errors": [raw_text]
            }
            
        # Pass 2: Structure & Validate
        structured_json = structure_raw_text_to_json(raw_text, doc_type)
        is_valid, errors, cleaned_data = validate_extracted_data(structured_json, doc_type)
        
        return {
            "success": True,
            "doc_type": doc_type,
            "extracted_data": cleaned_data,
            "raw_text": raw_text,
            "is_valid": is_valid,
            "validation_errors": errors
        }
        
    except Exception as e:
        return {
            "success": False,
            "doc_type": "unknown",
            "extracted_data": {},
            "raw_text": "",
            "is_valid": False,
            "validation_errors": [f"Pipeline exception: {str(e)}"]
        }