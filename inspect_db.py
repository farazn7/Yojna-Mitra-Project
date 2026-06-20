import sys

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    print("❌ Missing dependency: psycopg2")
    print("Install it with:")
    print("  py -m pip install psycopg2-binary")
    sys.exit(1)


def inspect_database(limit=15):
    # Connect to the local Postgres container
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="mysecretpassword",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor(cursor_factory=RealDictCursor)

    # 1. Show all schemas and tables in the database
    print("📚 Database overview")
    print("=" * 70)
    cur.execute("""
        SELECT schema_name
        FROM information_schema.schemata
        ORDER BY schema_name;
    """)
    print("Schemas:")
    for row in cur.fetchall():
        print(f"  - {row['schema_name']}")

    cur.execute("""
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_type = 'BASE TABLE'
        ORDER BY table_schema, table_name;
    """)
    print("\nTables:")
    for row in cur.fetchall():
        print(f"  - {row['table_schema']}.{row['table_name']}")
    print("\n" + "=" * 70)

    # 2. Inspect government_schemes if it exists
    cur.execute("""
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name = 'government_schemes'
        ) AS exists_flag;
    """)
    table_exists = cur.fetchone()['exists_flag']

    if not table_exists:
        print("❌ Table 'public.government_schemes' does not exist.")
        cur.close()
        conn.close()
        return

    print("\n🔎 Inspecting public.government_schemes")
    print("=" * 70)

    # 3. Total records
    cur.execute("SELECT COUNT(*) AS total FROM public.government_schemes;")
    total_records = cur.fetchone()['total']
    print(f"📊 Total records: {total_records}")

    if total_records == 0:
        print("❌ No records found in the table.")
        cur.close()
        conn.close()
        return

    # 4. Show schema for the table
    cur.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'government_schemes'
        ORDER BY ordinal_position;
    """)
    print("\n🔧 Columns:")
    for row in cur.fetchall():
        print(f"  - {row['column_name']}: {row['data_type']}")

    # 5. Show first rows with readable previews
    # Some existing databases may not have portal_url yet, so inspect the columns first
    cur.execute("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'government_schemes';
    """)
    existing_columns = {row['column_name'] for row in cur.fetchall()}

    select_columns = [
        'id',
        'scheme_name',
        'documents_needed',
        'LEFT(details, 300) AS details_preview',
        'LEFT(eligibility_rules, 300) AS eligibility_preview',
        'min_age',
        'max_age',
        'max_income',
        'is_differently_abled',
        'is_women_only'
    ]

    if 'portal_url' in existing_columns:
        select_columns.insert(2, 'portal_url')

    query = f"""
        SELECT
            {',\n            '.join(select_columns)},
            vector_dims(embedding) AS embedding_dims,
            LEFT(embedding::text, 90) AS embedding_preview
        FROM public.government_schemes
        ORDER BY id
        LIMIT %s;
    """
    cur.execute(query, (limit,))

    rows = cur.fetchall()
    print(f"\n🔎 First {len(rows)} rows (limit={limit}):")
    for row in rows:
        print("\n" + "-" * 70)
        for key in row.keys():
            value = row[key]
            if value is None:
                value = "NULL"
            elif isinstance(value, str) and len(value) > 250:
                value = value[:247] + "..."
            print(f"  {key}: {value}")
    print("\n" + "=" * 70)

    cur.close()
    conn.close()


if __name__ == "__main__":
    inspect_database()