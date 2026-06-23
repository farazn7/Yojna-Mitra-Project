import psycopg2
import ollama

def test_semantic_search():
    # 1. Connect to Postgres
    conn = psycopg2.connect(
        dbname="postgres", 
        user="postgres",
        password="mysecretpassword",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()

    # 2. Define a test query
    user_query = "My husband passed away recently, and I am struggling to manage our family farm alone in the village. I need financial support to buy crop seeds for this season."
    print(f" Searching for: '{user_query}'\n")

    # 3. Convert the query into a vector using Ollama
    response = ollama.embeddings(
        model='nomic-embed-text', 
        prompt=user_query
    )
    query_vector = response['embedding']

    # 4. Perform the Vector Search in Postgres!
    # The <=> operator calculates the cosine distance between the user's vector and the scheme's vector.
    # A lower distance means a closer semantic match.
    cur.execute("""
        SELECT scheme_name, (embedding <=> %s::vector) AS distance
        FROM government_schemes
        ORDER BY distance ASC
        LIMIT 5;
    """, (query_vector,))

    # 5. Fetch and print results
    results = cur.fetchall()
    print(" Top 5 Semantic Matches:")
    for rank, row in enumerate(results, 1):
        scheme_name = row[0]
        distance = row[1]
        print(f"{rank}. {scheme_name} (Distance: {distance:.4f})")

    cur.close()
    conn.close()

if __name__ == "__main__":
    test_semantic_search()