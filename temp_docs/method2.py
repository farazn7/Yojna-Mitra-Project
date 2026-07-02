import tempfile
import os

# 1. Create a temporary file that deletes itself when closed
with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
    # 2. Write the downloaded bytes from DigiLocker/Discord into this file
    temp_pdf.write(downloaded_document_bytes)
    file_path = temp_pdf.name  
    
    print(f"File temporarily saved at: {file_path}")

# 3. Extract text from 'file_path' and update the PostgreSQL schema
# ... extraction logic ...

# 4. Pass 'file_path' to Playwright to upload it
# ... playwright logic ...

# 5. Destroy the file manually when done
os.remove(file_path)
print("File destroyed.")