import json
import re
import ollama
from typing import Dict, List

# Documents we CAN handle via photo/scan upload and Vision LLM extraction
SCANNABLE_DOC_TYPES = [
    "aadhaar",
    "pan_card",
    "voter_id",
    "income_certificate",
    "caste_certificate",
    "community_certificate",
    "college_id",
    "ration_card",
    "passport",
    "driving_license",
    "birth_certificate",
    "marriage_certificate",
    "bank_passbook",
    "land_record",
    "age_proof",
    "residential_certificate",
    "photograph"
]

# Documents that typically must be obtained from offices or submitted physically/manually
NON_SCANNABLE = [
    "project_report",
    "quotation",
    "affidavit",
    "employer_certificate",
    "sterilization_certificate",
    "no_male_child_certificate"
]

def extract_and_print_thoughts(node_name: str, raw_response: str) -> str:
    """Extracts <think> tags, prints them to terminal immediately, and returns clean text."""
    match = re.search(r'<think>(.*?)</think>', raw_response, flags=re.DOTALL | re.IGNORECASE)
    
    if match:
        thoughts = match.group(1).strip()
        clean_text = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL | re.IGNORECASE).strip()
        
        print(f"\n [AGENT THOUGHTS: {node_name}]", flush=True)
        print(f"\033[90m{thoughts}\033[0m", flush=True) 
        print("-" * 50 + "\n", flush=True)
        
        if not clean_text:
            return ""
        return clean_text
        
    return raw_response.strip()

def parse_required_documents(documents_needed_text: str, scheme_name: str = "") -> Dict[str, List[str]]:
    """
    Uses Llama 3.1 to parse unstructured natural language `documents_needed` text into structured lists.
    
    Returns:
        {
            "scannable": ["aadhaar", "income_certificate", "caste_certificate"],
            "manual": ["Marriage Invitation", "Project Report with GST quotation"]
        }
    """
    clean_text = documents_needed_text.strip() if documents_needed_text else ""
    # Strip BOM or empty placeholder characters common in some JSON extractions
    clean_text = clean_text.replace("\ufeff", "").strip()
    
    if not clean_text:
        # Fallback if no explicit document text was provided in the database
        prompt = f"""You are determining standard required documents for an Indian government scheme named: "{scheme_name}".

Based on typical Indian welfare schemes of this nature, list the essential required documents.
Classify each document into ONE of two categories:

SCANNABLE (user can photograph and upload):
Valid labels ONLY from this exact list: {', '.join(SCANNABLE_DOC_TYPES)}

MANUAL (user must obtain from an office or prepare manually):
Return the descriptive document name as-is.

Output ONLY valid JSON in this exact format:
{{"scannable": ["aadhaar", "income_certificate"], "manual": []}}"""
    else:
        prompt = f"""You are parsing a list of required documents for an Indian government scheme ({scheme_name}).

Input text:
{clean_text}

Classify each document into ONE of two categories:

SCANNABLE (user can photograph and upload):
Valid labels ONLY from this exact list: {', '.join(SCANNABLE_DOC_TYPES)}

MANUAL (user must obtain from an office — cannot be photographed for our extraction purposes):
Return the original document name as-is.

Rules:
- "Identity proof i.e. Aadhaar card / Voter ID" or "Proof of identity (Aadhaar/Voter ID)" → pick "aadhaar" (primary ID)
- "Passport size photo" or "Photographs" → "photograph"
- "Proof of age" or "Date of birth" → "birth_certificate" (or "age_proof" if broader)
- "Community Certificate" and "Caste Certificate" are the SAME → "caste_certificate"
- Skip file format/size instructions like "*file size should be less than 200kb" or "*file type should be PDF"
- If a document says "or" between two options, pick the most common primary option
- "Bank Pass Book" or "Bank account details" → "bank_passbook"
- Deduplicate any repeated document labels

Output ONLY valid JSON inside markdown block or as raw JSON string in this exact format:
{{"scannable": ["aadhaar", "income_certificate"], "manual": ["Marriage Invitation"]}}"""

    try:
        response = ollama.chat(
            model='llama3.1',
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0}
        )
        
        raw_output = response['message']['content']
        cleaned_output = extract_and_print_thoughts("DOC REQUIREMENTS PARSER", raw_output)
        
        # Strip out markdown code block backticks if present
        if "```json" in cleaned_output:
            cleaned_output = cleaned_output.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned_output:
            cleaned_output = cleaned_output.split("```")[1].split("```")[0].strip()
            
        parsed_result = json.loads(cleaned_output)
        
        # Validate and clean up scannable tags against our known list
        scannable = []
        for item in parsed_result.get("scannable", []):
            label = str(item).lower().strip()
            if label in SCANNABLE_DOC_TYPES and label not in scannable:
                scannable.append(label)
                
        manual = [str(item).strip() for item in parsed_result.get("manual", []) if str(item).strip()]
        
        return {
            "scannable": scannable,
            "manual": manual
        }
        
    except Exception as e:
        print(f"[Doc Requirements Error] Failed to parse document requirements: {e}")
        # Safe fallback: require Aadhaar as universal baseline identity proof if parsing fails
        return {
            "scannable": ["aadhaar"],
            "manual": [clean_text] if clean_text else []
        }
