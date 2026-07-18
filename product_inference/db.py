import psycopg2
import json
import difflib
import re
from psycopg2.extras import RealDictCursor

# Using the standard configuration for the persistent container
DB_PARAMS = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "mysecretpassword",
    "host": "localhost",
    "port": "5432"
}

def init_db():
    """Initializes the database tables for user profiles and the PII Vault."""
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    
    # 1. User Profiles Table (Existing)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            id SERIAL PRIMARY KEY,
            platform_id VARCHAR(255) UNIQUE NOT NULL,
            username VARCHAR(255),
            current_state VARCHAR(50) DEFAULT 'START',
            profile_data JSONB DEFAULT '{}'::jsonb,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # 2. PII Vault Table (NEW)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS pii_vault (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(255) UNIQUE NOT NULL,
            canonical_data JSONB NOT NULL DEFAULT '{}'::jsonb,
            source_documents JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW(),
            expires_at TIMESTAMP NOT NULL,
            CONSTRAINT fk_vault_user FOREIGN KEY (user_id) 
                REFERENCES user_profiles(platform_id) ON DELETE CASCADE
        );
    """)
    
    # 3. Index for fast expiry lookups
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_vault_user_expiry ON pii_vault(user_id, expires_at);
    """)
    
    conn.commit()
    cur.close()
    conn.close()
    print("✅ Database schemas initialized (including PII Vault).")

def get_or_create_user(platform_id: str, username: str):
    """Fetches user record or creates one if it is their first time connecting."""
    conn = psycopg2.connect(**DB_PARAMS, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    
    query = """
        INSERT INTO user_profiles (platform_id, username, current_state, last_active)
        VALUES (%s, %s, 'START', CURRENT_TIMESTAMP)
        ON CONFLICT (platform_id) 
        DO UPDATE SET last_active = CURRENT_TIMESTAMP
        RETURNING platform_id, username, current_state, profile_data;
    """
    cur.execute(query, (str(platform_id), username))
    user = cur.fetchone()
    
    conn.commit()
    cur.close()
    conn.close()
    return user

def update_user_state(platform_id: str, next_state: str, updated_profile: dict = None):
    """Updates the conversation state machine and appends new keys to the profile JSONB object."""
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    
    if updated_profile is not None:
        query = """
            UPDATE user_profiles 
            SET current_state = %s, profile_data = profile_data || %s::jsonb 
            WHERE platform_id = %s;
        """
        cur.execute(query, (next_state, json.dumps(updated_profile), str(platform_id)))
    else:
        query = "UPDATE user_profiles SET current_state = %s WHERE platform_id = %s;"
        cur.execute(query, (next_state, str(platform_id)))
        
    conn.commit()
    cur.close()
    conn.close()

# ---------------------------------------------------------
# NEW PII VAULT HELPER FUNCTIONS
# ---------------------------------------------------------

def check_vault_cache(user_id: str, document_type: str) -> dict | None:
    """Check if we already have unexpired data from this SPECIFIC document type."""
    conn = psycopg2.connect(**DB_PARAMS, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    
    cur.execute("""
        SELECT canonical_data, source_documents 
        FROM pii_vault
        WHERE user_id = %s 
          AND expires_at > NOW()
          AND source_documents ? %s
    """, (str(user_id), document_type))
    
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row

def get_vault_data(user_id: str) -> dict | None:
    """Get ALL canonical data for Playwright form-filling. Ghost-filtered by expiry."""
    conn = psycopg2.connect(**DB_PARAMS, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    
    cur.execute("""
        SELECT canonical_data FROM pii_vault
        WHERE user_id = %s AND expires_at > NOW()
    """, (str(user_id),))
    
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row['canonical_data'] if row else None

def upsert_vault(user_id: str, document_type: str, extracted_fields: dict, source_meta: dict, ttl_hours: int = 24):
    """Insert or MERGE new document data into the user's vault row and restart TTL clock."""
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    
    cur.execute("""
        INSERT INTO pii_vault (user_id, canonical_data, source_documents, expires_at)
        VALUES (%s, %s::jsonb, %s::jsonb, NOW() + INTERVAL '%s hours')
        ON CONFLICT (user_id) DO UPDATE SET
            canonical_data = pii_vault.canonical_data || EXCLUDED.canonical_data,
            source_documents = pii_vault.source_documents || EXCLUDED.source_documents,
            updated_at = NOW(),
            expires_at = EXCLUDED.expires_at
    """, (
        str(user_id),
        json.dumps(extracted_fields),
        json.dumps({document_type: source_meta}),
        ttl_hours
    ))
    
    conn.commit()
    cur.close()
    conn.close()

def purge_expired_vault() -> int:
    """The garbage collector. Physically wipes expired PII from disk."""
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    
    cur.execute("DELETE FROM pii_vault WHERE expires_at <= NOW()")
    deleted = cur.rowcount
    
    conn.commit()
    cur.close()
    conn.close()
    return deleted

def get_scheme_documents_needed(scheme_name: str) -> str:
    """Queries PostgreSQL government_schemes table to fetch exact documents_needed text."""
    conn = psycopg2.connect(**DB_PARAMS, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    
    # Check exact or ILIKE match
    cur.execute("""
        SELECT documents_needed FROM government_schemes
        WHERE scheme_name ILIKE %s
        LIMIT 1
    """, (f"%{scheme_name.strip()}%",))
    
    row = cur.fetchone()
    cur.close()
    conn.close()
    
    return row['documents_needed'] if row and row.get('documents_needed') else ""

def get_all_scheme_names() -> list[str]:
    """Returns a list of all scheme names currently stored in government_schemes."""
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        cur.execute("SELECT scheme_name FROM government_schemes;")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [r[0] for r in rows if r and r[0]]
    except Exception as e:
        print(f"[DB Error] Could not fetch scheme names: {e}")
        return []

def find_similar_scheme_name(query: str, threshold: float = 0.25) -> str:
    """Finds the closest matching scheme name from government_schemes using SQL trigram similarity or Python difflib fallback."""
    if not query or not query.strip():
        return ""
    
    clean_q = query.strip()
    clean_q = re.sub(r'(?i).*\b(apply for|application for|register for|avail of|help with applying for|applying for|apply to)\s+', '', clean_q).strip(' .?!')
    if not clean_q:
        clean_q = query.strip()
        
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
            cur.execute("""
                SELECT scheme_name, similarity(scheme_name, %s) AS score
                FROM government_schemes
                WHERE similarity(scheme_name, %s) > %s
                   OR scheme_name ILIKE %s
                ORDER BY score DESC
                LIMIT 1;
            """, (clean_q, clean_q, threshold, f"%{clean_q}%"))
            row = cur.fetchone()
            if row and row[0]:
                cur.close()
                conn.close()
                return row[0]
        except Exception:
            conn.rollback()
            
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[DB Error in find_similar_scheme_name SQL check] {e}")
        
    all_schemes = get_all_scheme_names()
    if not all_schemes:
        return ""
        
    for s in all_schemes:
        if s.lower() in clean_q.lower() or clean_q.lower() in s.lower():
            return s
            
    matches = difflib.get_close_matches(clean_q, all_schemes, n=1, cutoff=threshold)
    if matches:
        return matches[0]
        
    q_words = set(w.lower() for w in re.findall(r'\w+', clean_q) if len(w) > 3)
    best_scheme = ""
    best_score = 0
    for s in all_schemes:
        s_words = set(w.lower() for w in re.findall(r'\w+', s) if len(w) > 3)
        if q_words and s_words:
            overlap = len(q_words.intersection(s_words))
    return best_scheme


def get_scheme_application_info(scheme_name: str) -> dict | None:
    """Fetches application_mode, application_form_url, portal_url, and application_process for a scheme."""
    try:
        conn = psycopg2.connect(**DB_PARAMS, cursor_factory=RealDictCursor)
        cur = conn.cursor()
        cur.execute("""
            SELECT scheme_name, portal_url, application_mode, application_form_url, application_process
            FROM government_schemes
            WHERE LOWER(scheme_name) = LOWER(%s) OR scheme_name ILIKE %s
            LIMIT 1;
        """, (scheme_name.strip(), f"%{scheme_name.strip()}%"))
        row = cur.fetchone()
        cur.close()
        conn.close()
        return dict(row) if row else None
    except Exception as e:
        print(f"[DB Error in get_scheme_application_info] {e}")
        return None


def update_scheme_application_info(scheme_name: str, mode: str, form_url: str, process: str):
    """Updates application_mode, application_form_url, and application_process for a scheme."""
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("""
            UPDATE government_schemes
            SET application_mode = %s,
                application_form_url = %s,
                application_process = %s
            WHERE LOWER(scheme_name) = LOWER(%s) OR scheme_name ILIKE %s;
        """, (mode, form_url, process, scheme_name.strip(), f"%{scheme_name.strip()}%"))
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[DB Error in update_scheme_application_info] {e}")