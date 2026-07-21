import psycopg2
from product_inference.db import DB_PARAMS

def migrate():
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        
        # Drop the foreign key constraint first
        cur.execute("ALTER TABLE pii_vault DROP CONSTRAINT IF EXISTS fk_vault_user;")
        
        # 1. Update user_profiles
        cur.execute("""
            UPDATE user_profiles 
            SET platform_id = 'discord_' || platform_id 
            WHERE platform_id NOT LIKE 'discord_%' AND platform_id NOT LIKE 'telegram_%';
        """)
        users_updated = cur.rowcount
        
        # 2. Update pii_vault
        cur.execute("""
            UPDATE pii_vault 
            SET user_id = 'discord_' || user_id 
            WHERE user_id NOT LIKE 'discord_%' AND user_id NOT LIKE 'telegram_%';
        """)
        vault_updated = cur.rowcount
        
        # Re-add the foreign key constraint with ON UPDATE CASCADE
        cur.execute("""
            ALTER TABLE pii_vault 
            ADD CONSTRAINT fk_vault_user 
            FOREIGN KEY (user_id) REFERENCES user_profiles(platform_id) 
            ON DELETE CASCADE ON UPDATE CASCADE;
        """)
        
        conn.commit()
        cur.close()
        conn.close()
        
        print(f"Migration successful: {users_updated} user profiles and {vault_updated} vault records updated to discord_ prefix.")
    except Exception as e:
        print(f"Migration failed: {e}")

if __name__ == "__main__":
    migrate()
