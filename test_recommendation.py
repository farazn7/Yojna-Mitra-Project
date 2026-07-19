from core_inference.hybrid_rag import run_yojana_pipeline

def test_recommendation():
    profile = {} # Empty profile
    query = "I am an unemployed youth looking to start a small business. Are there any schemes or loans to help me get capital or employment generation support?"
    
    print("Testing scheme recommendation with top_k=7...")
    response, schemes = run_yojana_pipeline(profile, query)
    
    print("\n\n=== RESPONSE ===\n")
    print(response)
    print("\n\n=== SCHEMES FETCHED ===\n")
    print(schemes)

if __name__ == "__main__":
    test_recommendation()
