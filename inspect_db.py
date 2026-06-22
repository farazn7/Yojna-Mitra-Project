import sys

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    print("❌ Missing dependency: psycopg2")
    print("Install it with: pip install psycopg2-binary")
    sys.exit(1)

def inspect_database(limit=5):
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="mysecretpassword",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor(cursor_factory=RealDictCursor)

    # 1. Fetch all tables strictly in the public schema
    cur.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        ORDER BY table_name;
    """)
    tables = [row['table_name'] for row in cur.fetchall()]

    print("📚 Database Overview (public schema)")
    print("=" * 70)
    if not tables:
        print("❌ No tables found in the 'public' schema yet.")
        print("Run your initialization/migration scripts to seed your tables.")
        cur.close()
        conn.close()
        return
        
    for t in tables:
        print(f"  - public.{t}")
    print("=" * 70)

    # 2. Dynamically inspect every table found
    for table in tables:
        print(f"\n🔎 Inspecting public.{table}")
        print("=" * 70)

        # Get Total Count
        cur.execute(f'SELECT COUNT(*) AS total FROM public."{table}";')
        total_records = cur.fetchone()['total']
        print(f"📊 Total records: {total_records}")

        # Get Columns & Types
        cur.execute(f"""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_schema = 'public' AND table_name = '{table}'
            ORDER BY ordinal_position;
        """, (table,))
        columns = cur.fetchall()
        
        print("\n🔧 Columns:")
        for col in columns:
            print(f"  - {col['column_name']}: {col['data_type']}")

        if total_records == 0:
            print("  ℹ️ Table is currently empty.")
            continue

        # Fetch raw rows and handle string truncation safely in Python
        cur.execute(f'SELECT * FROM public."{table}" LIMIT %s;', (limit,))
        rows = cur.fetchall()

        print(f"\n🔎 First {len(rows)} rows (limit={limit}):")
        for row in rows:
            print("\n" + "-" * 70)
            for key, value in row.items():
                # Format previews safely without crashing on vector or jsonb types
                if value is None:
                    display_val = "NULL"
                elif isinstance(value, (dict, list)):
                    import json
                    display_val = json.dumps(value)
                else:
                    display_val = str(value)

                # Safe character truncation
                if len(display_val) > 250:
                    display_val = display_val[:247] + "..."
                    
                print(f"  {key}: {display_val}")
        print("\n" + "=" * 70)

    cur.close()
    conn.close()

if __name__ == "__main__":
    inspect_database()