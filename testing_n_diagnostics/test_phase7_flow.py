import os
import sys
import io
import asyncio
import json

# Ensure UTF-8 output on Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core_inference.graph import graph_app
import product_inference.db as db

def run_simulation():
    print("=" * 70)
    print(" 🚀 YOJANA MITRA: PHASE 7 DOCUMENT COLLECTION & HITL SIMULATION")
    print("=" * 70)
    
    test_user_id = "test_user_phase7_1002"
    config = {"configurable": {"thread_id": test_user_id}}
    
    # 1. Setup mock user profile as PROFILE_COMPLETE so we skip onboarding questionnaire
    print(f"\n[Step 1] Setting up mock user profile in PostgreSQL for `{test_user_id}`...")
    db.get_or_create_user(test_user_id, "DiscordUser")
    mock_profile = {
        "gender": "Female",
        "age": 21,
        "income": 120000,
        "caste": "OBC",
        "residence": "Urban",
        "marital_status": "Single",
        "differently_abled": False
    }
    db.update_user_state(test_user_id, "PROFILE_COMPLETE", mock_profile)
    print("✅ User Profile set to PROFILE_COMPLETE.")
    
    # 2. User initiates application for a specific scheme
    target_scheme = "Moovalur Ramamirtham Ammaiyar Ninaivu Marriage Assistance Scheme-1"
    print(f"\n[Step 2] User says: 'I want to apply for {target_scheme}'...")
    print("-" * 70)
    
    output_1 = graph_app.invoke(
        {
            "messages": [{"role": "user", "content": f"I want to apply for {target_scheme} please."}],
            "user_id": test_user_id
        },
        config=config
    )
    
    print("\n[AI RESPONSE TO USER]")
    print(output_1.get("response", "No response."))
    print("-" * 70)
    
    # Check internal LangGraph state right now
    state_1 = graph_app.get_state(config)
    awaiting_doc = state_1.values.get("awaiting_document", "")
    pending_docs = state_1.values.get("pending_documents", [])
    collected_docs = state_1.values.get("collected_documents", [])
    
    print(f"\n📊 [LANGGRAPH STATE SNAPSHOT]")
    print(f"   • Target Scheme      : {state_1.values.get('target_scheme')}")
    print(f"   • Pending Documents  : {pending_docs}")
    print(f"   • Collected Documents: {collected_docs}")
    print(f"   • Currently Awaiting : **{awaiting_doc}**")
    print("=" * 70)
    
    if not awaiting_doc:
        print("⚠️ Graph did not pause for a document. Check intent classification or checklist.")
        return

    # 3. Simulate bot.py intercepting the requested document and saving to PII Vault
    print(f"\n[Step 3] Simulating bot.py: User uploads a valid `{awaiting_doc}` image/scan...")
    # Mock extracted fields that Vision LLM would produce
    mock_extracted = {
        "document_type": awaiting_doc,
        "verified_holder": "Anitha M",
        "status": "VALID"
    }
    db.upsert_vault(
        user_id=test_user_id,
        document_type=awaiting_doc,
        extracted_fields=mock_extracted,
        source_meta={"filename": f"mock_{awaiting_doc}.jpg", "file_path": f"./temp_documents/{test_user_id}/mock_{awaiting_doc}.jpg"},
        ttl_hours=24
    )
    print(f"✅ `{awaiting_doc}` successfully extracted & saved to PII Vault (`temp_documents/{test_user_id}/`).")
    
    # 4. Resume LangGraph with [DOC_RECEIVED: doc_type] signal (just like bot.py does!)
    print(f"\n[Step 4] bot.py resumes LangGraph with `[DOC_RECEIVED: {awaiting_doc}]` signal...")
    print("-" * 70)
    
    output_2 = graph_app.invoke(
        {
            "messages": [{"role": "user", "content": f"[DOC_RECEIVED: {awaiting_doc}]"}],
            "user_id": test_user_id
        },
        config=config
    )
    
    print("\n[AI RESPONSE TO USER]")
    print(output_2.get("response", "No response."))
    print("-" * 70)
    
    # Check updated LangGraph state
    state_2 = graph_app.get_state(config)
    awaiting_doc_2 = state_2.values.get("awaiting_document", "")
    collected_docs_2 = state_2.values.get("collected_documents", [])
    
    print(f"\n📊 [UPDATED LANGGRAPH STATE SNAPSHOT]")
    print(f"   • Pending Documents  : {state_2.values.get('pending_documents')}")
    print(f"   • Collected Documents: {collected_docs_2}")
    print(f"   • Currently Awaiting : **{awaiting_doc_2}**")
    print("=" * 70)
    print("\n🎉 PHASE 7 SIMULATION COMPLETE! The graph dynamically drove the checklist and paused for HITL input exactly as designed.")

if __name__ == "__main__":
    run_simulation()
