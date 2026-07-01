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
    current_intent: str                      # CHIT_CHAT | SCHEME_QUERY | PROFILE_UPDATE
    user_id: str                             # Discord Snowflake ID
    response: str                            # Final message to pass back to Discord

# ==========================================
# 2. GRAPH NODES (WORKERS)
# ==========================================

def load_user_profile(state: ConversationState) -> dict:
    """Entry node: Syncs state with the PostgreSQL profile table."""
    user_id = state["user_id"]
    # Fallback default configuration if database fetch fails
    onboarding_step = "START"
    user_profile = {}
    
    try:
        # Utilize your existing db entry logic
        user_record = db.get_or_create_user(user_id, "DiscordUser")
        if user_record:
            # Assuming user_record returns a dict or row mapping columns
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

    # --- FULL 14-STATE PROFILER MATRIX ---
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

    # Dual-Write Execution: Keep backend database updated
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
        
    latest_query = state["messages"][-1].content

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
            options={"temperature": 0.0}  # Lock down deterministic classification
        )
        raw_output = response['message']['content']
        # Extract the thoughts, print them to the terminal, and get the clean intent word!
        cleaned_output = extract_and_print_thoughts("INTENT CLASSIFIER", raw_output)
        
        predicted_intent = cleaned_output.strip().upper()
        if predicted_intent not in ["SCHEME_QUERY", "PROFILE_UPDATE", "CHIT_CHAT"]:
            predicted_intent = "SCHEME_QUERY"
    except Exception as e:
        print(f"[Graph Error] Intent classification step failed: {e}")
        predicted_intent = "SCHEME_QUERY"

    return {"current_intent": predicted_intent}


def handle_chit_chat(state: ConversationState) -> dict:
    """Executes context-aware chitchat using structural profile variables and conversation history."""
    # Build a descriptive persona overview using the flat DB parameters
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

    # Reconstruct sliding history payload for Ollama
    ollama_messages = [{"role": "system", "content": system_prompt}]
    for msg in state["messages"]:
        # Handles LangGraph's dynamic message classes cleanly
        role = "user" if msg.type == "human" else "assistant"
        ollama_messages.append({"role": role, "content": msg.content})

    try:
        response = ollama.chat(
            model='llama3.1',
            messages=ollama_messages
        )
        # Use the extractor to print thoughts and clean the reply!
        raw_reply = response['message']['content']
        reply = extract_and_print_thoughts("CHIT CHAT", raw_reply)
        
    except Exception as e:
        print(f"[Graph Error] Exception encountered inside Chit Chat: {e}")
        reply = "I'm having trouble connecting to my chat module right now. Please try again in a moment!"

    return {"response": reply}

def summarize_conversation(state: ConversationState) -> dict:
    """Condenses old messages into a summary and removes them to save VRAM."""
    summary = state.get("summary", "")
    messages = state.get("messages", [])

    # Keep the last 4 messages for immediate context, summarize the rest
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
        new_summary = summary # Fallback to old summary if crash occurs

    # LangGraph's safe deletion protocol: Return RemoveMessage objects matching the old IDs
    delete_messages = [RemoveMessage(id=m.id) for m in messages_to_summarize]

    return {
        "summary": new_summary,
        "messages": delete_messages
    }

def handle_scheme_query(state: ConversationState) -> dict:
    """Interfaces with your production pgvector engine using conversation logs for context continuity."""
    user_query = state["messages"][-1].content
    profile = state.get("user_profile", {})
    
    # Convert history into a clean sequential structure for your hybrid search engine pipeline
    formatted_history = []
    for msg in state["messages"][:-1]:  # Exclude current query
        role = "user" if msg.type == "human" else "assistant"
        formatted_history.append({"role": role, "content": msg.content})

    try:
        # Expecting modification in run_yojana_pipeline to accept history context
        # and output a tuple containing the textual response and schema identities
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
        "last_discussed_schemes": schemes_fetched  # Cached for sequential index indexing followups
    }


def handle_profile_update(state: ConversationState) -> dict:
    """Enables direct profile overrides and total tracking resets."""
    user_query = state["messages"][-1].content.lower()
    user_id = state["user_id"]
    profile = dict(state.get("user_profile", {}))
    
    # Clean structural catch for hard reset commands
    if "start over" in user_query or "reset" in user_query:
        db.update_user_state(user_id, "START", {})
        return {
            "onboarding_step": "START",
            "user_profile": {},
            "response": "Your saved profile data has been wiped. Let's restart. What is your gender?"
        }

    # Basic programmatic field override fallback handler
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
        # Strip out the think tags so json.loads doesn't crash!
        cleaned_output = extract_and_print_thoughts("PROFILE UPDATE", raw_output)
        
        updated_profile = json.loads(cleaned_output.strip())
        db.update_user_state(user_id, "PROFILE_COMPLETE", updated_profile)
        return {
            "user_profile": updated_profile,
            "response": "Got it! I've updated your profile parameter adjustments successfully."
        }
    except Exception as e:
        return {"response": "I couldn't process that update structural pattern. Could you try specifying it differently?"}


def append_response(state: ConversationState) -> dict:
    """Terminal node: Safely passes the response message to the state."""
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
    if intent == "SCHEME_QUERY":
        return "handle_scheme_query"
    elif intent == "PROFILE_UPDATE":
        return "handle_profile_update"
    return "handle_chit_chat"

def should_summarize(state: ConversationState) -> str:
    """Evaluates if the memory window has breached the 8-message limit."""
    if len(state["messages"]) > 8:
        return "summarize_conversation"
    return "__end__"  # Returns standard string to map to LangGraph's END

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

# FIXED: Explicit routing path mapping for the summarizer check
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
        "handle_scheme_query": "handle_scheme_query",
        "handle_profile_update": "handle_profile_update",
        "handle_chit_chat": "handle_chit_chat"
    }
)

builder.add_edge("handle_scheme_query", "append_response")
builder.add_edge("handle_profile_update", "append_response")
builder.add_edge("handle_chit_chat", "append_response")


# ==========================================
# 5. DATABASE CONNECTION & CHECKPOINTER
# ==========================================
DB_CONN_STRING = "postgresql://postgres:mysecretpassword@localhost:5432/postgres"

# Create a persistent connection pool with AUTOCOMMIT ENABLED
pool = ConnectionPool(
    conninfo=DB_CONN_STRING,
    kwargs={"autocommit": True} 
)

# Pass the pool directly into the PostgresSaver
checkpointer = PostgresSaver(pool)

# Safely auto-build state tracking tables inside the database container
checkpointer.setup()

# Final application run compilation targets
graph_app = builder.compile(checkpointer=checkpointer)