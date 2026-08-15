import re
import os
import json
import sys

# Portal text is routinely not Latin script (a scheme portal may render entirely in Devanagari or
# Tamil), and this module prints element labels and the model's own reasoning. A Windows console
# defaults to cp1252, where printing such a string raises UnicodeEncodeError and takes the whole run
# down — reproduced live on a button labelled "जमा करें". Logging must never be able to do that.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import ollama
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, RemoveMessage
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

# Import your existing modules
# Make sure your project paths match these import targets
import product_inference.db as db
import core_inference.hybrid_rag as hybrid_rag

def extract_and_print_thoughts(node_name: str, raw_response: str) -> str:
    """Extracts <think> tags, prints them to the terminal immediately, and returns clean text."""
    match = re.search(r'<think>(.*?)</think>', raw_response, flags=re.DOTALL | re.IGNORECASE)
    
    if match:
        thoughts = match.group(1).strip()
        clean_text = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL | re.IGNORECASE).strip()
        
        print(f"\n [AGENT THOUGHTS: {node_name}]", flush=True)
        print(f"\033[90m{thoughts}\033[0m", flush=True) 
        print("-" * 50 + "\n", flush=True)
        
        # SAFE GUARDRAIL: If the model only thought and didn't answer, return a fallback
        if not clean_text:
            return "I've analyzed the schemes matching your profile, but I ran out of room to format the response. Could you please try asking your question again?"
            
        return clean_text
    
    # Fallback if no think tags exist but text is empty
    if not raw_response.strip():
        return "Greetings. I processed your request but encountered an empty response generation. Let's try that again."
        
    return raw_response.strip()

# ==========================================
# 1. GRAPH STATE DEFINITION
# ==========================================
class ConversationState(TypedDict):
    messages: Annotated[list, add_messages]  # STM buffer holding chat history
    summary: str
    onboarding_step: str                     # Tracks current onboarding state
    user_profile: dict                       # Flat JSONB profile data from DB
    last_discussed_schemes: list             # Persists listed scheme names for indexing
    current_intent: str                      # CHIT_CHAT | SCHEME_QUERY | PROFILE_UPDATE | DOC_RECEIVED | APPLY_SCHEME | SKIP_DOCUMENT
    user_id: str                             # Discord Snowflake ID
    response: str                            # Final message to pass back to Discord
    # ── Document Collection State ──
    target_scheme: str                       # Name of scheme user wants to apply for
    scheme_for_pending_docs: str             # Tracks which scheme the current pending_documents array belongs to
    pending_documents: list                  # ["aadhaar", "income_certificate", "land_record"]
    collected_documents: list                # ["aadhaar"] — docs already in vault
    skipped_documents: list                  #  NEW: ["pan_card"] — docs user explicitly skipped
    awaiting_document: str                   # "income_certificate" — what we're waiting for RIGHT NOW
    user_input_scheme: str                   # Raw scheme input when fuzzy clarifying
    vault_snapshot: dict                     # Latest canonical_data from pii_vault
    # ── Phase 8 Automation State ──
    automation_status: str                   # "idle" | "running" | "hitl_paused" | "awaiting_confirm" | "complete" | "error"
    automation_session_id: str               # Browser session ID
    application_mode: str                    # "online" | "pdf_form" | "physical_only" | "unknown"
    application_form_url: str                # Direct URL to form or PDF download
    document_vault: dict                     # doc_type -> verified file path, forwarded to AutomationState for FILE_UPLOAD classification
    automation_unresolved_fields: list       # Unresolved field metadata carried from a paused turn into the next resume turn
    automation_ask_counts: dict              # field identity -> times the portal has re-asked it after the citizen answered (loop bound)
    automation_page_memory: dict             # screens visited, controls observed to move backwards, and what each screen offered on arrival; survives HITL resumes
    automation_hitl_answers: dict            # every answer the citizen has given this automation run, accumulated

# ==========================================
# 2. GRAPH NODES (WORKERS)
# ==========================================

def load_user_profile(state: ConversationState) -> dict:
    """Entry node: Syncs state with the PostgreSQL profile table."""
    user_id = state["user_id"]
    onboarding_step = "START"
    user_profile = {}
    
    try:
        platform = user_id.split("_")[0].capitalize() if "_" in user_id else "Unknown"
        user_record = db.get_or_create_user(user_id, f"{platform}User")
        if user_record:
            onboarding_step = user_record.get("current_state", "START")
            user_profile = user_record.get("profile_data", {})
    except Exception as e:
        print(f"[Graph Error] Failed to sync profile for user {user_id}: {e}")
        
    return {
        "onboarding_step": onboarding_step,
        "user_profile": user_profile
    }


def is_profile_complete(profile: dict) -> bool:
    """Checks if the minimum required profile fields are present."""
    required = {"gender", "age", "income", "caste", "occupation"}
    return bool(profile) and required.issubset(profile.keys())

def request_profile(state: ConversationState) -> dict:
    """Asks the user to fill out their profile using the platform-specific form."""
    return {
        "response": (
            "⚠️ **Profile Required**\n\n"
            "To apply for a scheme or get personalized recommendations, I need to know a bit about you first.\n\n"
            "Please run **/profile** to fill out a quick form. It takes less than a minute and lets me personalize everything for you!"
        )
    }


def classify_intent(state: ConversationState) -> dict:
    """Evaluates incoming queries to dynamically route fully onboarded users."""
    if not state["messages"]:
        return {"current_intent": "CHIT_CHAT"}

    latest_query = state["messages"][-1].content.strip()

    # Fast-track check for synthetic document reception signal from bot.py
    if latest_query.startswith("[DOC_RECEIVED:"):
        return {"current_intent": "DOC_RECEIVED"}

    # Automation being paused/running must not short-circuit classification outright —
    # doing so meant ANY message (e.g. "actually apply for a different scheme instead", "cancel")
    # was blindly forced into AUTOMATION_RESPONSE with zero real intent check. Real classification
    # still runs below; automation_was_active + the AUTOMATION_RESPONSE category (added to the
    # prompt) let the model decide whether this message is answering the pending automation
    # question or asking for something else. If the citizen overrides, the stale automation
    # session is cleaned up below rather than left dangling.
    automation_was_active = state.get("automation_status") in ("hitl_paused", "awaiting_confirm", "running")

    # Fast-track check if answering YES to scheme clarification
    if state.get("current_intent") == "CLARIFY_SCHEME_NAME" and latest_query.lower() in ["yes", "y", "yeah", "sure", "ok", "proceed", "start"]:
        return {"current_intent": "APPLY_SCHEME", "target_scheme": state.get("target_scheme", "")}

    # Context Injection
    last_schemes = state.get("last_discussed_schemes", [])
    awaiting_doc = state.get("awaiting_document", "none")
    current_target = state.get("target_scheme", "none")

    context_str = (
        f"Recently Discussed Schemes: {last_schemes}\n"
        f"Currently Selected Scheme: {current_target}\n"
        f"Currently Awaiting Document Upload: {awaiting_doc}\n"
    )
    if automation_was_active:
        context_str += (
            f"Automation Status: {state.get('automation_status')}\n"
            f"Pending Automation Question (my last message to the citizen): {state.get('response', '')}\n"
        )

    intent_enum = "SCHEME_QUERY | APPLY_SCHEME | PROFILE_UPDATE | SKIP_DOCUMENT | CHIT_CHAT"
    valid_intents = ["SCHEME_QUERY", "APPLY_SCHEME", "PROFILE_UPDATE", "SKIP_DOCUMENT", "CHIT_CHAT", "DOC_RECEIVED"]
    automation_intent_def = ""
    if automation_was_active:
        intent_enum = "SCHEME_QUERY | APPLY_SCHEME | PROFILE_UPDATE | SKIP_DOCUMENT | CHIT_CHAT | AUTOMATION_RESPONSE"
        valid_intents = valid_intents + ["AUTOMATION_RESPONSE"]
        automation_intent_def = (
            "- AUTOMATION_RESPONSE: ONLY valid because Automation Status above shows automation is active. "
            "Use this when the user's message is directly answering the Pending Automation Question above "
            "(a verification code, \"CONFIRM\", a requested value like a state/district name) rather than "
            "asking for something new. If the user clearly wants something else instead (a different scheme, "
            "to cancel, an unrelated question), classify their actual intent instead — do NOT default to "
            "AUTOMATION_RESPONSE just because automation happens to be active.\n"
        )

    system_prompt = (
        "You are the intent router for Yojana Mitra, an AI assistant for Indian government welfare schemes.\n"
        f"Context:\n{context_str}\n"
        "Given the user's message and the context above, output ONLY a valid JSON object with two keys:\n"
        "{\n"
        f"  \"intent\": \"<one of: {intent_enum}>\",\n"
        "  \"target_scheme\": \"<exact scheme name if intent is APPLY_SCHEME, else null>\"\n"
        "}\n\n"
        "Intent definitions:\n"
        "- SCHEME_QUERY: User is asking about eligibility, benefits, or searching for schemes.\n"
        "- APPLY_SCHEME: User wants to apply for or register for a specific scheme. (e.g. 'I want to apply', 'apply for book bank').\n"
        "- PROFILE_UPDATE: User wants to change their profile details (age, income, caste, etc.) or reset.\n"
        "- SKIP_DOCUMENT: User wants to skip the current document upload (e.g., 'skip', 'next', 'I don't have it'). ONLY select this if Currently Awaiting Document Upload is not 'none'.\n"
        "- CHIT_CHAT: Greetings, thanks, general conversation.\n"
        f"{automation_intent_def}\n"
        "IMPORTANT for APPLY_SCHEME:\n"
        "- If the user mentions a scheme by name (even abbreviated), set target_scheme to your best match from the Recently Discussed Schemes list.\n"
        "- If the user says something like 'apply for the second one', use the Recently Discussed Schemes list to resolve it to the exact string.\n"
        "- If the user just says 'I want to apply' without specifying, set target_scheme to the Currently Selected Scheme if one exists, otherwise null.\n\n"
        "Output ONLY the JSON object. Do not use markdown blocks, sentences, or punctuation."
    )

    predicted_intent = "CHIT_CHAT"
    target_scheme_raw = None
    classification_failed = False

    try:
        response = ollama.chat(
            model='llama3.1',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": latest_query}
            ],
            options={"temperature": 0.0}
        )
        raw_output = response['message']['content']
        cleaned_output = extract_and_print_thoughts("INTENT CLASSIFIER", raw_output)

        # Strip potential markdown formatting
        cleaned_output = cleaned_output.strip()
        if cleaned_output.startswith("```json"):
            cleaned_output = cleaned_output[7:]
        if cleaned_output.startswith("```"):
            cleaned_output = cleaned_output[3:]
        if cleaned_output.endswith("```"):
            cleaned_output = cleaned_output[:-3]

        parsed_json = json.loads(cleaned_output.strip())
        predicted_intent = parsed_json.get("intent", "CHIT_CHAT").strip().upper()
        target_scheme_raw = parsed_json.get("target_scheme")

        if predicted_intent not in valid_intents:
            predicted_intent = "CHIT_CHAT"
    except Exception as e:
        print(f"[Graph Error] Intent classification step failed: {e}")
        classification_failed = True
        # An Ollama hiccup must not silently cancel a real, in-progress scheme application —
        # preserve today's safe default of treating the reply as an automation answer.
        predicted_intent = "AUTOMATION_RESPONSE" if automation_was_active else "CHIT_CHAT"

    # If automation was active but the citizen's real intent is something else, clean up
    # the stale automation session so it doesn't linger as "hitl_paused"/"running"/"awaiting_confirm"
    # forever and keep re-triggering this same gate on the next message.
    override_fields = {}
    if automation_was_active and predicted_intent != "AUTOMATION_RESPONSE":
        print(f"[Graph: classify_intent] Automation was '{state.get('automation_status')}' but citizen intent changed to '{predicted_intent}' — cancelling stale automation session.")
        try:
            from product_inference.browser_manager import close_session
            close_session(state.get("automation_session_id", ""))
        except Exception as close_err:
            print(f"[Graph: classify_intent] Could not close stale automation session: {close_err}")
        override_fields = {
            "automation_status": "idle",
            "automation_session_id": "",
            "automation_unresolved_fields": []
        }

    if classification_failed:
        return {"current_intent": predicted_intent, **override_fields}

    # If APPLY_SCHEME, verify the scheme name with fuzzy search if needed
    if predicted_intent == "APPLY_SCHEME" and target_scheme_raw and target_scheme_raw.lower() != "none":
        # If it hallucinated or abbreviated, find the closest DB match
        fuzzy_target = db.find_similar_scheme_name(target_scheme_raw, threshold=0.30)
        if fuzzy_target:
            current_target = state.get("target_scheme", "")
            is_new_scheme = bool(current_target and current_target != fuzzy_target)

            if fuzzy_target in last_schemes:
                resp = {"current_intent": "APPLY_SCHEME", "target_scheme": fuzzy_target}
            else:
                # Let's clarify to be safe
                resp = {"current_intent": "CLARIFY_SCHEME_NAME", "target_scheme": fuzzy_target, "user_input_scheme": target_scheme_raw}

            # Critical Bugfix: Prevent old documents bleeding into new schemes!
            if is_new_scheme:
                resp["pending_documents"] = []
                resp["collected_documents"] = []
                resp["skipped_documents"] = []
                resp["awaiting_document"] = ""

            return {**resp, **override_fields}
        else:
            # Could not find a match at all
            return {"current_intent": "CHIT_CHAT", **override_fields} # Fallback if we completely fail to resolve it

    if predicted_intent == "APPLY_SCHEME":
        return {"current_intent": "APPLY_SCHEME", "target_scheme": "", **override_fields}

    return {"current_intent": predicted_intent, **override_fields}


def handle_chit_chat(state: ConversationState) -> dict:
    profile = state.get("user_profile", {})
    profile_summary = ", ".join([f"{k}: {v}" for k, v in profile.items()])
    summary = state.get("summary", "")
    summary_text = f"\nPrevious Conversation Summary: {summary}" if summary else ""
    
    system_prompt = (
        "You are Yojana Mitra, a helpful AI assistant for government schemes.\n"
        f"User Profile Context: [{profile_summary}]. Utilize this info naturally if relevant.\n"
        f"{summary_text}\n"
        "Respond politely, concisely, and support code-mixed Hinglish naturally if the user initiates it."
    )

    ollama_messages = [{"role": "system", "content": system_prompt}]
    for msg in state["messages"]:
        role = "user" if msg.type == "human" else "assistant"
        ollama_messages.append({"role": role, "content": msg.content})

    try:
        response = ollama.chat(
            model='llama3.1',
            messages=ollama_messages
        )
        raw_reply = response['message']['content']
        reply = extract_and_print_thoughts("CHIT CHAT", raw_reply)
        
    except Exception as e:
        print(f"[Graph Error] Exception encountered inside Chit Chat: {e}")
        reply = "I'm having trouble connecting to my chat module right now. Please try again in a moment!"

    return {"response": reply}

def summarize_conversation(state: ConversationState) -> dict:
    summary = state.get("summary", "")
    messages = state.get("messages", [])

    messages_to_summarize = messages[:-4]
    
    prompt = (
        f"You are a memory module. Summarize the following conversation context. "
        f"If there is an existing summary, seamlessly integrate the new details into it.\n\n"
        f"Existing summary: {summary}\n\n"
        f"New conversation to summarize:\n"
    )
    
    for msg in messages_to_summarize:
        role = "User" if msg.type == "human" else "Yojana Mitra"
        prompt += f"{role}: {msg.content}\n"

    try:
        response = ollama.chat(
            model='llama3.1',
            messages=[{"role": "user", "content": prompt}]
        )
        raw_summary = response['message']['content']
        new_summary = extract_and_print_thoughts("MEMORY SUMMARIZER", raw_summary)
    except Exception as e:
        print(f"[Graph Error] Summarization failed: {e}")
        new_summary = summary

    delete_messages = [RemoveMessage(id=m.id) for m in messages_to_summarize]

    return {
        "summary": new_summary,
        "messages": delete_messages
    }

def handle_scheme_query(state: ConversationState) -> dict:
    user_query = state["messages"][-1].content
    profile = state.get("user_profile", {})
    
    formatted_history = []
    for msg in state["messages"][:-1]: 
        role = "user" if msg.type == "human" else "assistant"
        formatted_history.append({"role": role, "content": msg.content})

    try:
        reply_text, schemes_fetched = hybrid_rag.run_yojana_pipeline(
            profile_data=profile, 
            text=user_query,
            conversation_history=formatted_history
        )
        if not is_profile_complete(profile):
            reply_text += "\n\n---\n*💡 These results are general. Run **/profile** to get results filtered exactly to your age, income, and category!*"
    except Exception as e:
        print(f"[Graph Error] Core RAG execution failed: {e}")
        reply_text = "I ran into a problem scanning matching programs. Let me check my directory again."
        schemes_fetched = state.get("last_discussed_schemes", [])

    return {
        "response": reply_text,
        "last_discussed_schemes": schemes_fetched 
    }


def handle_profile_update(state: ConversationState) -> dict:
    user_query = state["messages"][-1].content.lower()
    user_id = state["user_id"]
    profile = dict(state.get("user_profile", {}))
    
    if "start over" in user_query or "reset" in user_query:
        db.update_user_state(user_id, "START", {})
        return {
            "onboarding_step": "START",
            "user_profile": {},
            "response": "Your saved profile data has been wiped. Let's restart. What is your gender?"
        }

    system_prompt = (
        f"Given this JSON profile: {json.dumps(profile)} and this user update request: '{user_query}', "
        "output the modified, complete JSON profile map. Change ONLY fields explicitly specified. "
        "Output valid raw JSON only, no markdown markers."
    )
    
    try:
        response = ollama.chat(
            model='llama3.1',
            messages=[{"role": "user", "content": system_prompt}]
        )
        raw_output = response['message']['content']
        cleaned_output = extract_and_print_thoughts("PROFILE UPDATE", raw_output)
        
        updated_profile = json.loads(cleaned_output.strip())
        db.update_user_state(user_id, "PROFILE_COMPLETE", updated_profile)
        return {
            "user_profile": updated_profile,
            "response": "Got it! I've updated your profile parameter adjustments successfully."
        }
    except Exception as e:
        return {"response": "I couldn't process that update structural pattern. Could you try specifying it differently?"}


def request_next_document(state: ConversationState) -> dict:
    """ UPDATED: Handles Cache Checks and Skip Logic dynamically."""
    user_id = state["user_id"]
    pending = list(state.get("pending_documents", []))
    collected = list(state.get("collected_documents", []))
    skipped = list(state.get("skipped_documents", []))
    target_scheme = state.get("target_scheme", "")
    intent = state.get("current_intent", "")
    awaiting = state.get("awaiting_document", "")
    tracked_scheme = state.get("scheme_for_pending_docs", "")
    
    if target_scheme and target_scheme != tracked_scheme:
        # User switched to a different scheme — reset doc state
        pending = []
        collected = []
        skipped = []
        awaiting = ""
    
    # 0. CLARIFY SCHEME NAME LOGIC: If intent is CLARIFY_SCHEME_NAME, ask Yes/No clarification immediately
    if intent == "CLARIFY_SCHEME_NAME":
        fuzzy_target = state.get("target_scheme", "")
        user_input = state.get("user_input_scheme", "")
        return {
            "target_scheme": fuzzy_target,
            "scheme_for_pending_docs": fuzzy_target,
            "response": (
                f"🔍 **Scheme Name Clarification**\n\n"
                f"You asked to apply for: **{user_input or 'your scheme'}**.\n"
                f"We found **{fuzzy_target}** in our active database.\n\n"
                f"Would you like to start the document upload checklist for **{fuzzy_target}** right now? *(Type Yes to apply, or No to search for other schemes)*"
            )
        }

    # 1. SKIP LOGIC: If intent was SKIP, log it and reset awaiting so we move to next
    if intent == "SKIP_DOCUMENT" and awaiting:
        print(f"[SKIP LOGIC] User chose to skip: {awaiting}")
        if awaiting not in skipped:
            skipped.append(awaiting)
        awaiting = ""

    # ALWAYS check the application mode first — do not trust stale checkpointed pending_documents
    # This prevents old docs from a previous scheme from bleeding into a new scheme request.
    if target_scheme:
        app_info = db.get_scheme_application_info(target_scheme) or {}
        mode = app_info.get("application_mode", "unknown")

        # If mode is NOT online, bypass document collection entirely — route straight to launch_automation
        # launch_automation will handle pdf_form, physical_only, and unknown with appropriate messages
        if mode != "online":
            print(f"[Graph] Bypassing doc collection for '{target_scheme}' — DB mode is '{mode}'. Routing to launch_automation.")
            return {
                "pending_documents": [],
                "collected_documents": [],
                "skipped_documents": [],
                "awaiting_document": "",
                "scheme_for_pending_docs": target_scheme,
                "current_intent": "LAUNCH_AUTOMATION"
            }

    # If first time asking for docs for this scheme (pending is empty):
    manual_docs = []
    if not pending and target_scheme:
            
        raw_docs_text = db.get_scheme_documents_needed(target_scheme)
        from core_inference.doc_requirements import parse_required_documents
        parsed = parse_required_documents(raw_docs_text, scheme_name=target_scheme) 
        pending = parsed.get("scannable", ["aadhaar"])
        manual_docs = parsed.get("manual", [])
    
    # 2. CACHE HIT LOGIC: Check which docs are already in the vault and verified physically on disk
    from product_inference.document_handler import find_document_file
    for doc_type in list(pending):
        if doc_type in skipped or doc_type in collected:
            if doc_type in collected:
                file_path = find_document_file(user_id, doc_type)
                if not file_path:
                    print(f"[CACHE MISS/EXPIRED ON DISK] {doc_type} record found but file missing on disk. Re-requesting.")
                    collected.remove(doc_type)
            continue
        # Only do DB check if we haven't skipped/collected it
        cached = db.check_vault_cache(user_id, doc_type)
        if cached:
            file_path = find_document_file(user_id, doc_type)
            if file_path:
                print(f"[CACHE HIT & VERIFIED] Found {doc_type} in vault and verified on disk. Skipping request.")
                collected.append(doc_type)
            else:
                print(f"[CACHE EXPIRED ON DISK] Found {doc_type} in vault DB but file missing on disk. Re-requesting.")
    
    # Find next uncollected AND unskipped document
    next_doc = None
    for doc_type in pending:
        if doc_type not in collected and doc_type not in skipped:
            next_doc = doc_type
            break
            
    vault_data = db.get_vault_data(user_id) or {}
    
    if next_doc is None:
        # Hard Gate: Ensure NO mandatory documents were skipped before Playwright confirmation
        if len(skipped) > 0:
            skipped_labels = ", ".join([s.replace("_", " ").title() for s in skipped])
            first_missing = skipped[0]
            # Move skipped back into uncollected state to enforce collection before form submission
            return {
                "pending_documents": pending,
                "collected_documents": collected,
                "skipped_documents": [], # Reset skipped so user is prompted again
                "awaiting_document": first_missing,
                "scheme_for_pending_docs": target_scheme,
                "vault_snapshot": vault_data,
                "response": (
                    f"⚠️ **Mandatory Documents Missing for Portal Submission**\n\n"
                    f"You previously skipped uploading: **{skipped_labels}**.\n"
                    "The official government portal requires these exact files to complete your application.\n"
                    f"Please upload a clear photo/scan of your **{first_missing.replace('_', ' ').title()}** now to proceed!"
                )
            }
            
        # ALL DOCS COLLECTED AND VERIFIED — route directly to launch_automation node!
        return {
            "pending_documents": pending,
            "collected_documents": collected,
            "skipped_documents": skipped,
            "awaiting_document": "",
            "scheme_for_pending_docs": target_scheme,
            "vault_snapshot": vault_data,
            "current_intent": "LAUNCH_AUTOMATION"
        }
        
    doc_label = next_doc.replace("_", " ").title()
    progress = f"({len(collected) + len(skipped)}/{len(pending)} processed)"
    
    manual_note = ""
    if manual_docs and (len(collected) + len(skipped)) == 0:
        manual_list_str = "\n".join([f"• {m}" for m in manual_docs])
        manual_note = f"\n\n*(Note: You will also need to keep the following offline/manual documents handy during final submission:\n{manual_list_str})*"
        
    return {
        "target_scheme": target_scheme,
        "pending_documents": pending,
        "collected_documents": collected,
        "skipped_documents": skipped,
        "awaiting_document": next_doc,
        "scheme_for_pending_docs": target_scheme,
        "vault_snapshot": vault_data,
        "response": (
            f"📄 **Document Required for {target_scheme or 'application'} {progress}:**\n\n"
            f"Please upload a clear photo/scan of your **{doc_label}**.\n"
            f"Make sure the full document is well-lit and readable so I can verify and secure it in your Vault.\n"
            f"*(If you don't have it right now, just type 'skip')*{manual_note}"
        )
    }


# The one wording of the final review gate. Both paths that can reach the gate must say the same
# thing: `launch_automation` (portal needed nothing from the citizen) and `handle_automation_response`
# (the citizen answered an OTP/HITL question first). These had drifted into two messages, and the
# resume path — the *more* common one, since any portal with an OTP goes through HITL — had lost the
# sentence stating that the halt before the submit button is deliberate. That sentence is the
# zero-auto-submit promise as the citizen experiences it, so it does not get to vary by code path.
FINAL_REVIEW_GATE_RESPONSE = (
    "[Final Review Gate] **Automated Application Form Filled Up to Final Review Gate!**\n\n"
    "In accordance with Citizen Safety Rules, I have halted right before the final submission button.\n"
    "Please review the attached portal screenshot carefully to ensure all details match your expectations.\n\n"
    "If everything looks correct, reply with **CONFIRM** to execute the final official submission!"
)


def launch_automation(state: ConversationState) -> dict:
    """Launches Phase 8 ReAct automation engine or provides physical/PDF guidelines per citizen preference."""
    target_scheme = state.get("target_scheme", "")
    user_id = state.get("user_id", "default_user")

    # Guard: if we somehow got here with no scheme, ask user to clarify
    if not target_scheme or not target_scheme.strip():
        return {
            "automation_status": "idle",
            "response": (
                "🔍 **Which scheme would you like to apply for?**\n\n"
                "I couldn't identify the exact scheme name. Please mention it by name "
                "(e.g. *'I want to apply for Book Bank Scheme'*) and I'll look it up for you!"
            )
        }

    vault_snapshot = state.get("vault_snapshot")
    if not vault_snapshot:
        try:
            vault_snapshot = db.get_vault_data(user_id) or {}
        except Exception as e:
            print(f"[Graph: launch_automation Notice] Could not fetch vault data from PostgreSQL: {e}")
            vault_snapshot = {}

    # 1. Read pre-computed application mode and links from DB
    app_info = db.get_scheme_application_info(target_scheme) or {}
    mode = app_info.get("application_mode", "unknown")
    portal_url = app_info.get("portal_url", "")
    
    # Parse application_links JSONB (pre-computed by classify_modes.py)
    app_links_raw = app_info.get("application_links") or {}
    if isinstance(app_links_raw, str):
        import json as _json
        try:
            app_links_raw = _json.loads(app_links_raw)
        except Exception:
            app_links_raw = {}
    
    online_link = app_links_raw.get("online_link", "") or ""
    pdf_links = app_links_raw.get("pdf_links", []) or []
    fallback_link = app_links_raw.get("fallback_link", "") or ""

    print(f"[Graph: launch_automation] Scheme: '{target_scheme}' | Mode: {mode.upper()} | Online: {online_link} | PDFs: {len(pdf_links)}")

    # 2. Handle PDF mode — send ALL PDF links to the user
    if mode == "pdf_form" and pdf_links:
        pdf_links_text = "\n".join([f"📎 [Download Form {i+1}]({url})" for i, url in enumerate(pdf_links)])
        fallback_text = f"\n\n**Official Scheme Page:** [View Details]({fallback_link or portal_url})" if (fallback_link or portal_url) else ""
        return {
            "application_mode": "pdf_form",
            "application_form_url": pdf_links[0],
            "automation_status": "idle",
            "response": (
                f"📋 **Application Form(s) for {target_scheme}**\n\n"
                "This scheme requires a physical application. Download the form(s) below, "
                "fill them using your verified details from your Yojana Mitra Vault, and submit at your local government office.\n\n"
                f"{pdf_links_text}"
                f"{fallback_text}\n\n"
                "📍 Submit your completed form at your nearest **e-Sevai Kendra**, **Taluk Office**, or the relevant departmental office."
            )
        }

    # 3. Handle physical_only mode
    if mode == "physical_only":
        ref_link = fallback_link or portal_url
        return {
            "application_mode": "physical_only",
            "application_form_url": ref_link,
            "automation_status": "idle",
            "response": (
                f"🏛️ **Physical Submission Required for {target_scheme}**\n\n"
                "This scheme only accepts in-person applications at government offices. "
                "No online portal or downloadable form is currently available.\n\n"
                + (f"**Official Reference:** [View Scheme Details]({ref_link})\n\n" if ref_link else "")
                + "📍 Please visit your nearest **e-Sevai Kendra** or the relevant departmental office "
                "with the documents verified in your Yojana Mitra Vault."
            )
        }

    # 4. Handle unknown mode — give user whatever info we have
    if mode == "unknown":
        ref_link = fallback_link or portal_url
        try:
            raw_docs = db.get_scheme_documents_needed(target_scheme) or ""
            docs_section = f"\n\n📋 **Documents you'll typically need:**\n{raw_docs.strip()}" if raw_docs.strip() else ""
        except Exception:
            docs_section = ""
        return {
            "application_mode": "unknown",
            "application_form_url": ref_link,
            "automation_status": "idle",
            "response": (
                f"🏛️ **How to Apply: {target_scheme}**\n\n"
                "This scheme does **not** have an online application portal. "
                "You'll need to apply in-person at your nearest government office."
                f"{docs_section}\n\n"
                + (f"📎 **Official Scheme Page:** [View Details]({ref_link})\n\n" if ref_link else "")
                + "📍 Please visit your nearest **e-Sevai Kendra**, **Taluk Office**, or the relevant departmental office to submit your application."
            )
        }


    # 5. Mode is ONLINE → Launch ReAct Automation Graph!
    if mode != "online":
        # Safety guard: never fall through into live browser automation for a scheme that
        # wasn't explicitly classified "online" (e.g. `pdf_form` with an empty `pdf_links`
        # list from missing/legacy `application_links` data — automation must not run in that gap).
        ref_link = fallback_link or portal_url
        return {
            "application_mode": mode,
            "application_form_url": ref_link,
            "automation_status": "idle",
            "response": (
                f"🏛️ **How to Apply: {target_scheme}**\n\n"
                f"I couldn't find a ready-to-use automated application path for this scheme (mode: `{mode}`)."
                + (f"\n\n📎 **Official Scheme Page:** [View Details]({ref_link})" if ref_link else "")
                + "\n\n📍 Please visit your nearest **e-Sevai Kendra**, **Taluk Office**, or the relevant departmental office for assistance."
            )
        }

    target_url = online_link or portal_url
    if not target_url:
        return {
            "automation_status": "error",
            "response": f"[WARN] Could not locate an active online submission link for '{target_scheme}'."
        }

    print(f"[Graph: launch_automation] Launching ReAct state machine across target: {target_url}")
    from product_inference.automation_graph import build_automation_graph
    auto_graph = build_automation_graph()

    merged_vault = dict(state.get("user_profile", {}))
    merged_vault.update(vault_snapshot)

    initial_auto_state = {
        "session_id": f"auto_{user_id}",
        "portal_url": target_url,
        "status": "perceiving",
        "user_vault": merged_vault,
        "document_vault": state.get("document_vault", {}),
        "actions_to_execute": [],
        "unresolved_questions": [],
        "history_signatures": [],
        "regressive_gates": {},
        "initial_gates": {}
    }

    try:
        auto_result = auto_graph.invoke(initial_auto_state)
        new_status = auto_result.get("status", "error")
        questions = auto_result.get("unresolved_questions", [])

        if new_status == "hitl_paused":
            q_text = "\n".join(questions) if questions else "Verification required on the portal."
            return {
                "automation_status": "hitl_paused",
                "automation_session_id": f"auto_{user_id}",
                "application_mode": mode,
                "application_form_url": target_url,
                "automation_unresolved_fields": auto_result.get("unresolved_fields", []),
                "automation_ask_counts": {}, "automation_page_memory": {}, "automation_hitl_answers": {},  # Fresh launch — start the re-ask bound clean.
                "automation_page_memory": _page_memory(auto_result),
                "response": f"[Portal Intercept] **Portal Intercepted for Human Verification**\n\n{q_text}"
            }
        elif new_status == "awaiting_final_confirmation":
            return {
                "automation_status": "awaiting_confirm",
                "automation_session_id": f"auto_{user_id}",
                "application_mode": mode,
                "application_form_url": target_url,
                "automation_unresolved_fields": [],
                "response": FINAL_REVIEW_GATE_RESPONSE
            }
        elif new_status == "form_complete":
            return {
                "automation_status": "complete",
                "automation_session_id": f"auto_{user_id}",
                "automation_unresolved_fields": [],
                "response": f"[SUCCESS] **Application Successfully Submitted!**\n\nYour application for **{target_scheme}** has been completed on the official portal."
            }
        else:
            return {
                "automation_status": "error",
                "automation_unresolved_fields": [],
                "response": f"[WARN] **Automation Notice:** The portal check resulted in status: `{new_status}`. Please check the portal manually at: {target_url}"
            }
    except Exception as e:
        print(f"[Graph: launch_automation Error] {e}")
        return {
            "automation_status": "error",
            "automation_unresolved_fields": [],
            "response": f"[ERROR] **Automation Error:** Encountered an unexpected issue while interacting with the portal: {e}"
        }


def _execute_confirmed_submission(session_id: str, state: ConversationState) -> dict:
    """
    Presses the portal's final submission control — the one action in the whole flow that cannot be
    undone, and the only one reached solely because the citizen typed CONFIRM on this turn.

    Three things have to hold for that press to actually complete an application, and each was missing:
      * the right control has to be found. It used to be located by English button text, which finds
        nothing on a portal reading "जमा करें" and could match the wrong button on the most
        consequential click in the flow. Which control submits is a reading of the screen, so the model
        makes it (llm_planner.identify_submit_control) and Python presses what it names — or nothing.
      * a native "are you sure?" dialog has to be accepted. The interceptor dismisses dialogs by
        default, which is correct in general but silently discarded the submission the citizen had just
        authorized. A one-shot, expiring authorization is armed here, in the CONFIRM path only.
      * the outcome has to be reported honestly. A screenshot of whatever the portal ended up showing
        is captured for the citizen's records, and a press that could not be made is said out loud
        rather than reported as a submission.
    """
    from product_inference.browser_manager import (
        get_active_page, run_pw, authorize_one_dialog
    )
    from product_inference.perception_engine import perceive_dom_state
    from product_inference.llm_planner import identify_submit_control

    active_page = run_pw(get_active_page, session_id)
    if not active_page or run_pw(active_page.is_closed):
        return {
            "automation_status": "error",
            "response": (
                "[WARN] **The portal session has closed**\n\nI could not press Submit because the browser "
                f"session ended. Nothing was submitted. You can finish at: {state.get('application_form_url', 'the scheme portal')}"
            )
        }

    perception = run_pw(perceive_dom_state, active_page)
    submit_id = identify_submit_control(perception)
    if submit_id is None:
        return {
            "automation_status": "awaiting_confirm",
            "response": (
                "[WARN] **I could not identify the submit control**\n\nRather than press the wrong button on "
                "the final screen, I have stopped. Nothing was submitted. The portal is still open at the "
                "review screen — you can press Submit yourself, or reply **CONFIRM** to let me try again."
            )
        }

    shot_path = os.path.join("screenshots", f"{session_id}_submitted.png")
    try:
        # Authorize the portal's own confirmation dialog, then press. Armed immediately before the
        # click and expiring on its own, so it can only ever cover this press.
        run_pw(authorize_one_dialog, session_id)
        run_pw(active_page.locator(f'[data-ym-id="{submit_id}"]').first.click, timeout=10000)
        run_pw(active_page.wait_for_timeout, 2500)
        os.makedirs("screenshots", exist_ok=True)
        run_pw(active_page.screenshot, path=shot_path, full_page=True)
    except Exception as e:
        print(f"[Confirmation Submit Notice] {e}")
        return {
            "automation_status": "error",
            "response": (
                f"[WARN] **The submission press failed**\n\n`{e}`\n\nI cannot confirm the application was "
                f"submitted. Please check the open portal window before trying again."
            )
        }

    # Say what actually happened. Announcing a submission the portal never made would leave a citizen
    # believing their application is filed when it is not — the worst failure this flow can produce,
    # and one that already occurred: the press was reported as success while the portal sat unchanged
    # on the review screen. Read from the page rather than matched against English phrases, which say
    # nothing on a portal written in Hindi or Tamil.
    from product_inference.llm_planner import page_confirms_submission
    try:
        page_text = run_pw(active_page.locator("body").inner_text, timeout=5000)
    except Exception:
        page_text = ""

    if not page_confirms_submission(page_text):
        return {
            "automation_status": "awaiting_confirm",
            "response": (
                "[WARN] **I pressed Submit but the portal has not confirmed it**\n\nThe page is not "
                "showing an acknowledgement, so I cannot tell you the application is filed. Please check "
                "the attached screenshot and the open portal window. Reply **CONFIRM** to press again if "
                "it is still waiting."
            )
        }

    return {
        "automation_status": "complete",
        "response": (
            "[SUCCESS] **Application Submitted**\n\nYour CONFIRM was received, the portal's final "
            "submission control was pressed, and the portal has confirmed receipt. The attached "
            "screenshot shows exactly what it displayed — please save any acknowledgement number for tracking."
        )
    }


def _page_memory(auto_result: dict) -> dict:
    """
    What the automation subgraph learned about the portal's own navigation, lifted out so it can
    survive a HITL pause. Each resume is a fresh subgraph invocation, so anything the ReAct loop
    observed — which screens it has already stood on, which navigation control on which screen
    turned out to move backwards, which controls a screen offered before anything was entered — is
    lost unless ConversationState carries it across the pause.

    `initial_gates` matters most of all here, and specifically because of this pause: the citizen's
    answer is written into the DOM by `_inject_hitl_answer` below, which is what unlocks the field's
    own confirm control. Losing the snapshot would make that control look like it had been there all
    along on the very cycle that has to prefer it — leaving the planner to pick by document order and
    press a captcha "Refresh" that discards the answer just given.
    """
    return {
        "history_signatures": list(auto_result.get("history_signatures") or []),
        "regressive_gates": {k: list(v) for k, v in (auto_result.get("regressive_gates") or {}).items()},
        "pressed_gates": {k: list(v) for k, v in (auto_result.get("pressed_gates") or {}).items()},
        "initial_gates": {k: list(v) for k, v in (auto_result.get("initial_gates") or {}).items()},
    }


def _hitl_field_identity(unresolved_fields: list) -> str:
    """
    Identity of the field a paused automation turn is asking about. The planner asks about exactly one
    field per turn (the first), so this is what the citizen's next reply will be attributed to. Used
    only to notice that the SAME field is being asked again.
    """
    if not unresolved_fields:
        return ""
    first = unresolved_fields[0] or {}
    return str(first.get("name") or first.get("label") or first.get("id") or "")


# How many times the citizen may answer the same field before we stop asking and hand the portal back
# to them. A wrong captcha or a mistyped OTP genuinely deserves retries; an unanswerable field must
# not be able to spin forever.
MAX_REPEAT_HITL_ASKS = 3


def _inject_hitl_answer(session_id: str, unresolved_fields: list, answer: str) -> bool:
    """
    Writes the citizen's reply straight into the element whose question they just answered.

    A portal challenge — an OTP, a captcha — is a live, single-use response, not durable vault data.
    Routing it through the vault and expecting the classifier to re-discover which field it belongs
    to is unreliable (observed live: the planner offered the citizen's mobile number for a field
    labelled "Enter Verification Code") and also unnecessary: we recorded which element prompted the
    question, so the answer's destination is a known fact rather than something to infer. Deciding
    *what* to ask stays a semantic judgment; delivering the answer to the field that asked is
    mechanics.
    """
    if not unresolved_fields or not answer:
        return False

    field = unresolved_fields[0] or {}
    el_id = field.get("id")
    if el_id is None:
        return False

    try:
        from product_inference.browser_manager import get_active_page, run_pw
        page = run_pw(get_active_page, session_id)
        if not page or run_pw(page.is_closed):
            return False

        locator = run_pw(page.locator, f'[data-ym-id="{el_id}"]')
        if run_pw(locator.count) == 0:
            return False
        target = locator.first

        # `data-ym-id` is re-assigned on every perception pass, so confirm this really is still the
        # element we asked about before typing the citizen's words into it.
        recorded_name = str(field.get("name") or "")
        if recorded_name:
            live_name = run_pw(target.get_attribute, "name") or ""
            if live_name != recorded_name:
                print(f"[HITL Inject] Element #{el_id} is now '{live_name}', not '{recorded_name}' — skipping direct injection.")
                return False

        # Only an element that actually accepts typed text may be filled — Playwright raises
        # otherwise ("Element is not an <input>, <textarea> or [contenteditable]", hit live when the
        # field that asked was a toggle). The perceived `type` recorded at question time is not a
        # safe proxy for this, so ask the live DOM: the set below is exactly the input types the HTML
        # spec gives no text value to type into. A toggle's answer is a yes/no reading that belongs to
        # the classifier (llm_planner.verify_checkbox_against_vault) and a file input needs a verified
        # document path, so both correctly fall out here rather than being special-cased.
        shape = run_pw(target.evaluate, """el => ({
            tag: el.tagName.toLowerCase(),
            type: (el.getAttribute('type') || 'text').toLowerCase(),
            editable: el.isContentEditable === true
        })""")
        NON_TEXT_INPUTS = {"checkbox", "radio", "file", "submit", "button", "reset", "image", "hidden", "range", "color"}
        fillable = (
            shape.get("editable")
            or shape.get("tag") == "textarea"
            or (shape.get("tag") == "input" and shape.get("type") not in NON_TEXT_INPUTS)
        )
        if not fillable:
            print(f"[HITL Inject] Element #{el_id} ({shape.get('tag')}/{shape.get('type')}) takes no typed value — "
                  f"leaving it to the classifier and the vault write-back.")
            return False

        run_pw(target.fill, str(answer))
        print(f"[HITL Inject] Wrote the citizen's answer into the field that asked for it ('{recorded_name or el_id}').")
        return True
    except Exception as e:
        print(f"[HITL Inject Notice] Could not write the answer into the field that asked: {e}")
        return False


def handle_automation_response(state: ConversationState) -> dict:
    """Handles citizen replies when the automation graph is paused for HITL verification or final CONFIRM."""
    user_id = state.get("user_id", "default_user")
    session_id = state.get("automation_session_id", f"auto_{user_id}")
    status = state.get("automation_status", "idle")
    latest_msg = state["messages"][-1].content.strip()

    from product_inference.automation_graph import build_automation_graph
    auto_graph = build_automation_graph()

    # 1. Handle Final Confirmation Gate
    if status == "awaiting_confirm":
        if latest_msg.upper() == "CONFIRM":
            print(f"[Graph: handle_automation_response] Citizen confirmed submission! Executing final button...")
            return _execute_confirmed_submission(session_id, state)
        else:
            return {
                "response": "[WARN] **Waiting for Confirmation**\n\nI am currently holding at the Final Review screen. Please reply exactly with **CONFIRM** when you are ready to submit, or type 'cancel' to abort."
            }

    # 2. Handle HITL Intercept (OTP, Captcha, missing PII answer)
    if status == "hitl_paused":
        print(f"[Graph: handle_automation_response] Injecting citizen input '{latest_msg}' into paused session...")
        merged_vault = dict(state.get("user_profile", {}))
        merged_vault.update(state.get("vault_snapshot", {}))

        # Every answer given so far this run, not just this turn's. Each resume builds a fresh vault
        # for a fresh subgraph invocation, so an answer written on one turn was gone by the next —
        # the citizen agreed to a consent box, the next cycle found nothing backing it and asked
        # again, and the re-ask bound eventually handed the portal back (observed live on
        # 'not_a_robot'). Text answers survived only because they also go into the field itself.
        hitl_answers = dict(state.get("automation_hitl_answers") or {})
        merged_vault.update(hitl_answers)

        merged_vault["hitl_input"] = latest_msg
        if re.match(r'^\d{4,8}$', latest_msg):
            merged_vault["otp"] = latest_msg

        # Also write the citizen's answer under the actual unresolved field's name,
        # mirroring hitl_intercept_node's Fix-6d logic (automation_graph.py) — the generic
        # "hitl_input"/"otp" keys above never match verify_field_against_vault's keyword-based
        # check for any non-OTP field (e.g. a "State" dropdown), which otherwise loops forever
        # re-asking the same question. Strictly additive: OTP mid-flow intercepts never populate
        # automation_unresolved_fields, so that path is unaffected.
        # The planner asks about exactly ONE field per paused turn (the first entry), so this reply
        # belongs to that field unambiguously — no need for the previous "only write back when exactly
        # one field is unresolved" guard, which silently skipped the write-back entirely whenever a
        # page had two things missing (observed live on the OTP step: nothing was ever saved, so the
        # same questions came back every cycle). `name` is the element's DOM name, a stable key the
        # next classification round reliably matches; `label` remains a fallback for older payloads.
        unresolved_fields = state.get("automation_unresolved_fields", [])
        if unresolved_fields:
            field_name = unresolved_fields[0].get("name") or unresolved_fields[0].get("label", "").lower()
            if field_name:
                merged_vault[field_name] = latest_msg
                hitl_answers[field_name] = latest_msg

        # Deliver the answer to the element that asked for it, so a live portal challenge does not
        # depend on the classifier re-deriving where it belongs. The vault write-back above still
        # happens: it is what makes a DURABLE field (a state, a name) match on later cycles and on
        # later schemes, whereas this injection is what makes a single-use challenge work at all.
        _inject_hitl_answer(session_id, unresolved_fields, latest_msg)

        resume_state = {
            "session_id": session_id,
            "portal_url": state.get("application_form_url", ""),
            "status": "perceiving",
            "user_vault": merged_vault,
            "document_vault": state.get("document_vault", {}),
            "actions_to_execute": [],
            "unresolved_questions": [],
            "retries": 0,
            # Carry forward what the loop already worked out about this portal's navigation.
            **(state.get("automation_page_memory") or {}),
        }

        try:
            auto_result = auto_graph.invoke(resume_state)
            new_status = auto_result.get("status", "error")
            questions = auto_result.get("unresolved_questions", [])

            if new_status == "hitl_paused":
                # *** Loop bound: never re-ask the same field indefinitely ***
                # A field the portal keeps rejecting (a mis-read captcha, a value the portal validates
                # differently than we do) would otherwise pause -> ask -> answer -> pause forever, since
                # each resume is a fresh graph invocation with no memory of the last one. Counting
                # asks here — in ConversationState, which survives across turns via the checkpointer
                # — is the only place that memory exists. This makes non-convergence structurally
                # impossible for ANY field rather than for the specific ones we happen to have seen.
                #
                # Tallied PER FIELD rather than as a consecutive streak: a page with two stubborn
                # fields alternates between them (observed live, OTP <-> captcha), which resets a
                # streak counter every turn and slips the bound entirely. A per-field tally cannot
                # be evaded by interleaving.
                new_unresolved = auto_result.get("unresolved_fields", [])
                asked_identity = _hitl_field_identity(unresolved_fields)
                still_asking = _hitl_field_identity(new_unresolved)
                ask_counts = dict(state.get("automation_ask_counts") or {})
                if asked_identity:
                    ask_counts[asked_identity] = ask_counts.get(asked_identity, 0) + 1

                if still_asking and ask_counts.get(still_asking, 0) >= MAX_REPEAT_HITL_ASKS:
                    from product_inference.browser_manager import close_session
                    label = (new_unresolved[0] or {}).get("label") or still_asking
                    print(f"[Graph: handle_automation_response] '{still_asking}' answered {ask_counts[still_asking]}x and still being asked — handing the portal back to the citizen.")
                    close_session(session_id)
                    return {
                        "automation_status": "error",
                        "automation_unresolved_fields": [],
                        "automation_ask_counts": {}, "automation_page_memory": {}, "automation_hitl_answers": {},
                        "response": (
                            f"[WARN] **I couldn't get past “{label}” on the portal**\n\n"
                            f"The portal kept asking for it again after each of your answers, so I've stopped "
                            f"rather than keep you in a loop.\n\n"
                            f"Everything I filled in so far is saved in your vault. You can finish this step "
                            f"yourself at: {state.get('application_form_url', 'the scheme portal')}"
                        )
                    }

                q_text = "\n".join(questions) if questions else "Further verification required."
                return {
                    "automation_status": "hitl_paused",
                    "automation_unresolved_fields": new_unresolved,
                    "automation_ask_counts": ask_counts,
                    "automation_page_memory": _page_memory(auto_result),
                    "automation_hitl_answers": hitl_answers,
                    "response": f"[Portal Intercept] **Verification Update**\n\n{q_text}"
                }
            elif new_status == "awaiting_final_confirmation":
                return {
                    "automation_status": "awaiting_confirm",
                    "automation_unresolved_fields": [],
                    "automation_ask_counts": {}, "automation_page_memory": {}, "automation_hitl_answers": {},
                    "response": FINAL_REVIEW_GATE_RESPONSE
                }
            elif new_status == "form_complete":
                return {"automation_status": "complete", "automation_unresolved_fields": [], "automation_ask_counts": {}, "automation_page_memory": {}, "automation_hitl_answers": {}, "response": "[SUCCESS] **Application Successfully Submitted!**"}
            else:
                return {"automation_status": "error", "automation_unresolved_fields": [], "automation_ask_counts": {}, "automation_page_memory": {}, "automation_hitl_answers": {}, "response": f"[WARN] **Portal Status Update:** `{new_status}`"}
        except Exception as e:
            return {"automation_status": "error", "automation_unresolved_fields": [], "automation_ask_counts": {}, "automation_page_memory": {}, "automation_hitl_answers": {}, "response": f"[ERROR] **Automation Error:** {e}"}

    return {"response": "No active automation session in progress."}


def append_response(state: ConversationState) -> dict:
    ai_msg = {"role": "assistant", "content": state["response"]}
    return {"messages": [ai_msg]}

# ==========================================
# 3. CONDITIONAL EDGE ROUTING ROUTINES
# ==========================================
def check_onboarding(state: ConversationState) -> str:
    return "classify_intent"

def route_intent(state: ConversationState) -> str:
    intent = state.get("current_intent", "CHIT_CHAT")
    profile = state.get("user_profile", {})

    if intent == "APPLY_SCHEME" and not is_profile_complete(profile):
        return "request_profile"

    if intent == "AUTOMATION_RESPONSE":
        return "handle_automation_response"
    elif intent == "LAUNCH_AUTOMATION":
        return "launch_automation"
    elif intent in ["DOC_RECEIVED", "APPLY_SCHEME", "SKIP_DOCUMENT", "CLARIFY_SCHEME_NAME"]:
        return "request_next_document"
    elif intent == "SCHEME_QUERY":
        return "handle_scheme_query"
    elif intent == "PROFILE_UPDATE":
        return "handle_profile_update"
    return "handle_chit_chat"

def should_summarize(state: ConversationState) -> str:
    if len(state["messages"]) > 8:
        return "summarize_conversation"
    return "__end__"

def check_document_collection_done(state: ConversationState) -> str:
    if state.get("current_intent") == "LAUNCH_AUTOMATION":
        return "launch_automation"
    return "append_response"

# ==========================================
# 4. GRAPH ASSEMBLY & PERSISTENCE COMPILATION
# ==========================================
builder = StateGraph(ConversationState)

# Node Registrations
builder.add_node("load_user_profile", load_user_profile)
builder.add_node("classify_intent", classify_intent)
builder.add_node("handle_chit_chat", handle_chit_chat)
builder.add_node("handle_scheme_query", handle_scheme_query)
builder.add_node("handle_profile_update", handle_profile_update)
builder.add_node("request_next_document", request_next_document)
builder.add_node("launch_automation", launch_automation)
builder.add_node("handle_automation_response", handle_automation_response)
builder.add_node("append_response", append_response)
builder.add_node("summarize_conversation", summarize_conversation)
builder.add_node("request_profile", request_profile)

# Linear and Conditional Wire mapping
builder.add_edge(START, "load_user_profile")
builder.add_edge("load_user_profile", "classify_intent")

builder.add_conditional_edges(
    "append_response", 
    should_summarize,
    {
        "summarize_conversation": "summarize_conversation",
        "__end__": END
    }
)

builder.add_edge("summarize_conversation", END)
builder.add_edge("request_profile", "append_response")

builder.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "request_profile": "request_profile",
        "request_next_document": "request_next_document",
        "handle_scheme_query": "handle_scheme_query",
        "handle_profile_update": "handle_profile_update",
        "handle_chit_chat": "handle_chit_chat",
        "launch_automation": "launch_automation",
        "handle_automation_response": "handle_automation_response"
    }
)

builder.add_conditional_edges(
    "request_next_document",
    check_document_collection_done,
    {
        "launch_automation": "launch_automation",
        "append_response": "append_response"
    }
)

builder.add_edge("launch_automation", "append_response")
builder.add_edge("handle_automation_response", "append_response")
builder.add_edge("handle_scheme_query", "append_response")
builder.add_edge("handle_profile_update", "append_response")
builder.add_edge("handle_chit_chat", "append_response")


# ==========================================
# 5. DATABASE CONNECTION & CHECKPOINTER
# ==========================================
DB_CONN_STRING = "postgresql://postgres:mysecretpassword@localhost:5432/postgres"

pool = ConnectionPool(
    conninfo=DB_CONN_STRING,
    kwargs={"autocommit": True} 
)

checkpointer = PostgresSaver(pool)
try:
    checkpointer.setup()
except Exception as e:
    print(f"[Checkpointer Setup Note] Skipping setup check (tables likely already initialized or locked): {e}")

graph_app = builder.compile(checkpointer=checkpointer)