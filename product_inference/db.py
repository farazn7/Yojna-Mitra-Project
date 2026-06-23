import psycopg2
import json
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
    """Initializes the database table for structured schema evaluation."""
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    
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
    conn.commit()
    cur.close()
    conn.close()

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