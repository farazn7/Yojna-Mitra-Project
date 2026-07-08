import psycopg2
import sys

def clear_pii_vault(user_id=None):
    """Deletes records from the PII Vault. If no user_id is given, it wipes the whole table."""
    try:
        conn = psycopg2.connect(
            dbname="postgres",
            user="postgres",
            password="mysecretpassword",
            host="localhost",
            port="5432"
        )
        cur = conn.cursor()

        if user_id:
            print(f"🧹 Wiping vault data for Discord ID: {user_id}...")
            cur.execute("DELETE FROM pii_vault WHERE user_id = %s;", (str(user_id),))
        else:
            print("🧹 Wiping the ENTIRE PII vault...")
            cur.execute("DELETE FROM pii_vault;")

        deleted = cur.rowcount
        conn.commit()
        
        cur.close()
        conn.close()
        
        print(f"✅ Success! Deleted {deleted} record(s) from the vault.")
        
    except Exception as e:
        print(f"❌ Database error: {e}")

if __name__ == "__main__":
    # If you run this script directly, it will wipe your specific test user
    target_user = "741229262954168371" 
    clear_pii_vault(target_user)
    
    # NOTE: If you ever want to wipe EVERYTHING in the vault to start totally fresh, 
    # just change the line above to: clear_pii_vault()