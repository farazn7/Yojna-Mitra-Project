import psycopg2

conn = psycopg2.connect(dbname="postgres", user="postgres", password="mysecretpassword", host="localhost", port="5432")
cur = conn.cursor()
cur.execute("DROP TABLE IF EXISTS government_schemes;")
conn.commit()
print("✅ Old government_schemes table destroyed. Ready for a fresh start.")
cur.close()
conn.close()