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
    
    if not clean_text or len(clean_text) < 5:
        # Graceful fallback: If no docs are provided by the DB, do NOT hallucinate them.
        return {
            "scannable": [],
            "manual": ["Please refer to the official scheme guidelines for the required document checklist."]
        }
        
    prompt = f"""You are a precise document extraction assistant for an Indian government scheme.

Scheme Name: "{scheme_name}"

Input text of required documents:
'''
{clean_text}
'''

Your task is to classify EACH required document mentioned in the input text into ONE of two categories:

1. SCANNABLE: Standard documents we can verify instantly via photo.
MUST use ONLY exact labels from this list: {', '.join(SCANNABLE_DOC_TYPES)}

2. MANUAL: Custom, unique, or un-scannable documents (e.g. project reports, custom certificates, forms).
Extract the exact descriptive phrase from the text.

CRITICAL RULES:
1. DO NOT HALLUCINATE. If "Aadhaar" or "Photograph" is NOT in the input text, DO NOT include them in your output.
2. If the input text is blank or only contains a single vague sentence, both arrays should be empty.
3. Map "Age Proof" or "Date of birth" -> "age_proof" or "birth_certificate".
4. Map "Identity proof" -> "aadhaar" (only if Aadhaar/Voter ID is explicitly mentioned as an example).

You must output ONLY valid JSON in this EXACT structure:
{{
  "reasoning": "Step 1: I see 'Death Certificate', this is not in the scannable list, so it goes to manual. Step 2: I see 'Registration Card', this goes to manual. Neither Aadhaar nor Photograph are mentioned, so I will NOT include them.",
  "scannable": ["label1", "label2"],
  "manual": ["exact phrase 1", "exact phrase 2"]
}}"""

    try:
        response = ollama.chat(
            model='llama3.1',
            messages=[{"role": "user", "content": prompt}],
            format='json',
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
        # Safe fallback: Do NOT hallucinate Aadhaar. Just return the raw text as manual.
        return {
            "scannable": [],
            "manual": [clean_text] if clean_text else []
        }


def evaluate_application_mode(sources_references: list, scheme_name: str = "") -> Dict:
    """
    Uses Llama 3.1 to intuitively analyze a scheme's Sources & References links
    and determine the application mode + extract ALL relevant links.
    
    This replaces the hardcoded keyword matching in classify_application_mode.
    Called at runtime when a user wants to apply — takes ~1-2 seconds.
    
    Args:
        sources_references: List of {"label": "...", "url": "..."} dicts from the DB
        scheme_name: Name of the scheme for context
    
    Returns:
        {
            "mode": "online" | "pdf_form" | "physical_only" | "unknown",
            "online_link": "https://..." or null,
            "pdf_links": ["https://...pdf", ...],
            "fallback_link": "https://..." (portal/guidelines link for reference)
        }
    """
    # If no sources at all, it's physical_only — no LLM needed
    if not sources_references:
        return {
            "mode": "physical_only",
            "online_link": None,
            "pdf_links": [],
            "fallback_link": None
        }
    
    # Format the links for the LLM prompt
    links_text = ""
    for i, item in enumerate(sources_references, 1):
        label = item.get("label", "Unknown")
        url = item.get("url", "")
        links_text += f"{i}. Label: \"{label}\" | URL: {url}\n"
    
    prompt = f"""You are analyzing the "Sources And References" links from an Indian government scheme page to determine how a citizen can apply for this scheme.

Scheme Name: "{scheme_name}"

Here are ALL the links found in the Sources & References section:
{links_text}

Your task: Determine the APPLICATION MODE by analyzing the link labels and URLs intuitively.

Rules for classification:
1. **online**: A link clearly leads to an ONLINE APPLICATION PORTAL where citizens can fill and submit forms digitally. Look for labels like "Apply Online", "Online Application", "Registration Portal", "e-Service", or URLs containing "apply", "register", "login", or online portal paths. Do NOT classify a link as "online" if it's just a scheme information page, guidelines, or eligibility criteria page.

2. **pdf_form**: One or more links point to downloadable PDF application forms that citizens need to print, fill, and submit physically. Look for URLs ending in ".pdf" or labels mentioning "Application Form", "Form Download". IMPORTANT: Exclude PDFs that are just "Guidelines", "Government Orders", "Notifications", or "Revised Rates" — those are NOT application forms.

3. **physical_only**: The links are only informational (guidelines, government orders, scheme details, notifications). No online portal or downloadable form is available. Citizens must visit a government office in person.

Respond with ONLY valid JSON in this exact format:
{{"mode": "pdf_form", "online_link": null, "pdf_links": ["https://example.com/form1.pdf", "https://example.com/form2.pdf"], "fallback_link": "https://example.com/guidelines"}}

- "mode": one of "online", "pdf_form", "physical_only"
- "online_link": the URL of the online application portal (null if mode is not "online")
- "pdf_links": list of ALL application form PDF URLs (empty list if none found). Include ALL relevant PDFs like English form, Tamil form, etc. Exclude guideline/notification PDFs.
- "fallback_link": the most useful informational link for the citizen (guidelines page, scheme page, etc.) — always provide one if available"""

    try:
        response = ollama.chat(
            model='llama3.1',
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0}
        )
        
        raw_output = response['message']['content']
        cleaned_output = extract_and_print_thoughts("APPLICATION MODE EVALUATOR", raw_output)
        
        # Strip markdown code block if present
        if "```json" in cleaned_output:
            cleaned_output = cleaned_output.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned_output:
            cleaned_output = cleaned_output.split("```")[1].split("```")[0].strip()
            
        result = json.loads(cleaned_output)
        
        # Validate and normalize
        valid_modes = {"online", "pdf_form", "physical_only", "unknown"}
        if result.get("mode") not in valid_modes:
            result["mode"] = "unknown"
        
        result.setdefault("online_link", None)
        result.setdefault("pdf_links", [])
        result.setdefault("fallback_link", None)
        
        # Ensure pdf_links is always a list
        if not isinstance(result["pdf_links"], list):
            result["pdf_links"] = [result["pdf_links"]] if result["pdf_links"] else []
        
        print(f"[App Mode Eval] {scheme_name} → {result['mode'].upper()} | online={result['online_link']} | pdfs={len(result['pdf_links'])}")
        return result
        
    except Exception as e:
        print(f"[App Mode Eval Error] Failed to evaluate application mode: {e}")
        # Safe fallback: return unknown so the system tries on-demand scrape
        return {
            "mode": "unknown",
            "online_link": None,
            "pdf_links": [],
            "fallback_link": sources_references[0].get("url") if sources_references else None
        }
