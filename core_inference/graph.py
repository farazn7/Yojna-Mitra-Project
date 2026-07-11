import re
import os
import json
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
        return "Namaste. I processed your request but encountered an empty response generation. Let's try that again."
        
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
    pending_documents: list                  # ["aadhaar", "income_certificate", "land_record"]
    collected_documents: list                # ["aadhaar"] — docs already in vault
    skipped_documents: list                  #  NEW: ["pan_card"] — docs user explicitly skipped
    awaiting_document: str                   # "income_certificate" — what we're waiting for RIGHT NOW
    user_input_scheme: str                   # Raw scheme input when fuzzy clarifying
    vault_snapshot: dict                     # Latest canonical_data from pii_vault

# ==========================================
# 2. GRAPH NODES (WORKERS)
# ==========================================

def load_user_profile(state: ConversationState) -> dict:
    """Entry node: Syncs state with the PostgreSQL profile table."""
    user_id = state["user_id"]
    onboarding_step = "START"
    user_profile = {}
    
    try:
        user_record = db.get_or_create_user(user_id, "DiscordUser")
        if user_record:
            onboarding_step = user_record.get("current_state", "START")
            user_profile = user_record.get("profile_data", {})
    except Exception as e:
        print(f"[Graph Error] Failed to sync profile for user {user_id}: {e}")
        
    return {
        "onboarding_step": onboarding_step,
        "user_profile": user_profile
    }


def onboarding_handler(state: ConversationState) -> dict:
    """Processes full questionnaire flow, validates numeric constraints, and executes dual-writes."""
    step = state.get("onboarding_step", "START")
    profile = dict(state.get("user_profile", {}))
    user_id = state["user_id"]
    
    user_input = ""
    if state["messages"]:
        latest_msg = state["messages"][-1]
        user_input = latest_msg.content.strip() if hasattr(latest_msg, "content") else latest_msg.get("content", "").strip()

    next_step = step
    response_text = ""

    if step == "START":
        next_step = "AWAITING_GENDER"
        response_text = "Welcome to Yojana Mitra! Let's get you set up to find matching schemes. 🇮🇳\n\n**Step 1:** What is your **Gender**? (e.g., Male, Female, Other)"
        
    elif step == "AWAITING_GENDER":
        profile["gender"] = user_input
        next_step = "AWAITING_AGE"
        response_text = "**Step 2:** What is your **Age**? (Numbers only)"
        
    elif step == "AWAITING_AGE":
        if user_input.isdigit():
            profile["age"] = int(user_input)
            next_step = "AWAITING_INCOME"
            response_text = "**Step 3:** What is your total annual family **Income** in INR? (Numbers only, e.g., 45000)"
        else:
            response_text = "❌ Please enter a valid numeric age:"
            
    elif step == "AWAITING_INCOME":
        if user_input.isdigit():
            profile["income"] = int(user_input)
            next_step = "AWAITING_CASTE"
            response_text = "**Step 4:** What is your **Caste** category? (e.g., General, OBC, SC, ST)"
        else:
            response_text = "❌ Please enter a valid number for income:"

    elif step == "AWAITING_CASTE":
        profile["caste"] = user_input
        next_step = "AWAITING_RESIDENCE"
        response_text = "**Step 5:** What is your area of **Residence**? (Rural / Urban)"

    elif step == "AWAITING_RESIDENCE":
        profile["residence"] = user_input
        next_step = "AWAITING_MARITAL"
        response_text = "**Step 6:** What is your **Marital Status**? (e.g., Single, Married, Widowed, Divorced)"

    elif step == "AWAITING_MARITAL":
        profile["marital_status"] = user_input
        next_step = "AWAITING_DISABLED"
        response_text = "**Step 7:** Are you **Differently Abled**? (Yes / No)"

    elif step == "AWAITING_DISABLED":
        is_disabled = user_input.lower() in ['yes', 'y', 'true']
        if is_disabled:
            profile["differently_abled"] = True
            next_step = "AWAITING_DISABILITY_PERC"
            response_text = "**Step 7b:** What is your **Disability Percentage**? (Enter number, or type 'None')"
        else:
            profile["differently_abled"] = False
            profile["disability_percentage"] = None
            next_step = "AWAITING_MINORITY"
            response_text = "**Step 8:** Do you belong to a **Minority** community? (Yes / No)"

    elif step == "AWAITING_DISABILITY_PERC":
        perc = int(user_input) if user_input.isdigit() else None
        profile["disability_percentage"] = perc
        next_step = "AWAITING_MINORITY"
        response_text = "**Step 8:** Do you belong to a **Minority** community? (Yes / No)"

    elif step == "AWAITING_MINORITY":
        profile["minority"] = user_input.lower() in ['yes', 'y', 'true']
        next_step = "AWAITING_BPL"
        response_text = "**Step 9:** Do you possess a **Below Poverty Line (BPL)** card? (Yes / No)"

    elif step == "AWAITING_BPL":
        profile["below_poverty_line"] = user_input.lower() in ['yes', 'y', 'true']
        next_step = "AWAITING_DISTRESS"
        response_text = "**Step 10:** Are you facing **Economic Distress**? (Yes / No)"

    elif step == "AWAITING_DISTRESS":
        profile["economic_distress"] = user_input.lower() in ['yes', 'y', 'true']
        next_step = "AWAITING_GOVT_EMP"
        response_text = "**Step 11:** Are you a **Government Employee**? (Yes / No)"

    elif step == "AWAITING_GOVT_EMP":
        profile["government_employee"] = user_input.lower() in ['yes', 'y', 'true']
        next_step = "AWAITING_OCCUPATION"
        response_text = "**Step 12:** What is your primary **Occupation**? (e.g., Farmer, Student, Artisan, Unemployed)"

    elif step == "AWAITING_OCCUPATION":
        profile["occupation"] = user_input
        next_step = "PROFILE_COMPLETE"
        
        response_text = (
            "🎉 **Yojana Mitra Profile Created Successfully!**\n"
            "The following structure is safely synced to PostgreSQL:\n\n"
            f"• **Gender:** {profile.get('gender')}\n"
            f"• **Age:** {profile.get('age')} years\n"
            f"• **Income:** ₹{profile.get('income', 0):,}\n"
            f"• **Caste:** {profile.get('caste')}\n"
            f"• **Residence:** {profile.get('residence')}\n"
            f"• **Marital Status:** {profile.get('marital_status')}\n"
            f"• **Differently Abled:** {profile.get('differently_abled')} (Perc: {profile.get('disability_percentage')}%)\n"
            f"• **Minority:** {profile.get('minority')}\n"
            f"• **BPL Status:** {profile.get('below_poverty_line')}\n"
            f"• **Economic Distress:** {profile.get('economic_distress')}\n"
            f"• **Govt Employee:** {profile.get('government_employee')}\n"
            f"• **Occupation:** {profile.get('occupation')}\n\n"
            "You are all set. Type any question now to search matching schemes!"
        )

    import product_inference.db as db
    try:
        db.update_user_state(user_id, next_step, profile)
    except Exception as e:
        print(f"[Graph Error] Onboarding database sync failed: {e}")

    return {
        "onboarding_step": next_step,
        "user_profile": profile,
        "response": response_text
    }


def classify_intent(state: ConversationState) -> dict:
    """Evaluates incoming queries to dynamically route fully onboarded users."""
    if not state["messages"]:
        return {"current_intent": "CHIT_CHAT"}
        
    latest_query = state["messages"][-1].content.strip()

    # Fast-track check for synthetic document reception signal from bot.py
    if latest_query.startswith("[DOC_RECEIVED:"):
        return {"current_intent": "DOC_RECEIVED"}

    # Fast-track check if answering YES to scheme clarification
    if state.get("current_intent") == "CLARIFY_SCHEME_NAME" and latest_query.lower() in ["yes", "y", "yeah", "sure", "ok", "proceed", "start"]:
        return {"current_intent": "APPLY_SCHEME", "target_scheme": state.get("target_scheme", "")}

    #  NEW: Fast-track check for skip command if we are currently waiting for a document
    if state.get("awaiting_document") and latest_query.lower() in ["skip", "next", "pass", "i don't have it"]:
        return {"current_intent": "SKIP_DOCUMENT"}

    # Fast-track check for application initiation intent
    if re.search(r'\b(apply|application|register|avail|fill form|start applying)\b', latest_query, re.IGNORECASE):
        target = state.get("target_scheme", "")
        last_schemes = state.get("last_discussed_schemes", [])
        if not target and last_schemes:
            for scheme in last_schemes:
                s_name = scheme if isinstance(scheme, str) else str(scheme)
                if s_name.lower() in latest_query.lower() or len(last_schemes) == 1:
                    target = s_name
                    break
            if not target and last_schemes:
                target = last_schemes[0] if isinstance(last_schemes[0], str) else str(last_schemes[0])
        
        if not target:
            all_db_schemes = db.get_all_scheme_names()
            for scheme_name in all_db_schemes:
                if scheme_name.lower() in latest_query.lower() or any(word.lower() in latest_query.lower() for word in scheme_name.split() if len(word) > 5):
                    target = scheme_name
                    break
        if not target:
            fuzzy_target = db.find_similar_scheme_name(latest_query, threshold=0.25)
            if fuzzy_target:
                return {"current_intent": "CLARIFY_SCHEME_NAME", "target_scheme": fuzzy_target, "user_input_scheme": latest_query}
        return {"current_intent": "APPLY_SCHEME", "target_scheme": target}

    system_prompt = (
        "You are the routing brain of Yojana Mitra, an assistant for Indian welfare programs.\n"
        "Categorize the user's latest statement into precisely one of these strings:\n"
        "1. SCHEME_QUERY: Asking about benefits, documentation requirements, or checking eligibility criteria.\n"
        "2. PROFILE_UPDATE: Intending to modify existing user profiles or starting onboarding fields from scratch.\n"
        "3. CHIT_CHAT: General talking, greetings, gratitude, or basic statements.\n\n"
        "Output ONLY the raw uppercase category string. Do not use markdown blocks, sentences, or punctuation."
    )

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
        
        predicted_intent = cleaned_output.strip().upper()
        if predicted_intent not in ["SCHEME_QUERY", "PROFILE_UPDATE", "CHIT_CHAT", "DOC_RECEIVED", "APPLY_SCHEME"]:
            predicted_intent = "SCHEME_QUERY"
    except Exception as e:
        print(f"[Graph Error] Intent classification step failed: {e}")
        predicted_intent = "SCHEME_QUERY"

    return {"current_intent": predicted_intent}


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
    
    # 0. CLARIFY SCHEME NAME LOGIC: If intent is CLARIFY_SCHEME_NAME, ask Yes/No clarification immediately
    if intent == "CLARIFY_SCHEME_NAME":
        fuzzy_target = state.get("target_scheme", "")
        user_input = state.get("user_input_scheme", "")
        return {
            "target_scheme": fuzzy_target,
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

    # If first time asking for docs for this scheme (pending is empty):
    manual_docs = []
    if not pending and target_scheme:
        raw_docs_text = db.get_scheme_documents_needed(target_scheme)
        from core_inference.doc_requirements import parse_required_documents
        parsed = parse_required_documents(raw_docs_text) 
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
                "vault_snapshot": vault_data,
                "response": (
                    f"⚠️ **Mandatory Documents Missing for Portal Submission**\n\n"
                    f"You previously skipped uploading: **{skipped_labels}**.\n"
                    "The official government portal requires these exact files to complete your application.\n"
                    f"Please upload a clear photo/scan of your **{first_missing.replace('_', ' ').title()}** now to proceed!"
                )
            }
            
        # ALL DOCS COLLECTED AND VERIFIED — proceed to application confirmation
        return {
            "pending_documents": pending,
            "collected_documents": collected,
            "skipped_documents": skipped,
            "awaiting_document": "",
            "vault_snapshot": vault_data,
            "response": (
                f"✅ **All required documents processed for {target_scheme or 'your scheme'}!**\n\n"
                f"I have secured and verified your files in your isolated Vault directory (`temp_documents/{user_id}/`).\n"
                "Your details are ready for automated form submission via the official government portal. Shall I initiate the application filling now?"
            )
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
        "vault_snapshot": vault_data,
        "response": (
            f"📄 **Document Required for {target_scheme or 'application'} {progress}:**\n\n"
            f"Please upload a clear photo/scan of your **{doc_label}**.\n"
            f"Make sure the full document is well-lit and readable so I can verify and secure it in your Vault.\n"
            f"*(If you don't have it right now, just type 'skip')*{manual_note}"
        )
    }


def append_response(state: ConversationState) -> dict:
    ai_msg = {"role": "assistant", "content": state["response"]}
    return {"messages": [ai_msg]}

# ==========================================
# 3. CONDITIONAL EDGE ROUTING ROUTINES
# ==========================================
def check_onboarding(state: ConversationState) -> str:
    if state.get("onboarding_step") != "PROFILE_COMPLETE":
        return "onboarding_handler"
    return "classify_intent"

def route_intent(state: ConversationState) -> str:
    intent = state.get("current_intent", "CHIT_CHAT")
    #  NEW: Add SKIP_DOCUMENT and CLARIFY_SCHEME_NAME to the array so it routes correctly
    if intent in ["DOC_RECEIVED", "APPLY_SCHEME", "SKIP_DOCUMENT", "CLARIFY_SCHEME_NAME"]:
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

# ==========================================
# 4. GRAPH ASSEMBLY & PERSISTENCE COMPILATION
# ==========================================
builder = StateGraph(ConversationState)

# Node Registrations
builder.add_node("load_user_profile", load_user_profile)
builder.add_node("onboarding_handler", onboarding_handler)
builder.add_node("classify_intent", classify_intent)
builder.add_node("handle_chit_chat", handle_chit_chat)
builder.add_node("handle_scheme_query", handle_scheme_query)
builder.add_node("handle_profile_update", handle_profile_update)
builder.add_node("request_next_document", request_next_document)
builder.add_node("append_response", append_response)
builder.add_node("summarize_conversation", summarize_conversation)

# Linear and Conditional Wire mapping
builder.add_edge(START, "load_user_profile")

builder.add_conditional_edges(
    "load_user_profile", 
    check_onboarding,
    {
        "onboarding_handler": "onboarding_handler",
        "classify_intent": "classify_intent"
    }
)

builder.add_conditional_edges(
    "append_response", 
    should_summarize,
    {
        "summarize_conversation": "summarize_conversation",
        "__end__": END
    }
)

builder.add_edge("summarize_conversation", END)
builder.add_edge("onboarding_handler", "append_response")

builder.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "request_next_document": "request_next_document",
        "handle_scheme_query": "handle_scheme_query",
        "handle_profile_update": "handle_profile_update",
        "handle_chit_chat": "handle_chit_chat"
    }
)

builder.add_edge("request_next_document", "append_response")
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