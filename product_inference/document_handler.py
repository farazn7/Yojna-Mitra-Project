import os
import struct
import io
import discord
from PIL import Image
import time

TEMP_DIR = "./temp_documents"
# Automatically create the folder if it doesn't exist
os.makedirs(TEMP_DIR, exist_ok=True)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB limit
ALLOWED_EXTENSIONS = {'.pdf', '.jpg', '.jpeg', '.png'}

# Hidden file headers to prevent malware disguised as PDFs/JPGs
MAGIC_BYTES = {
    '.pdf':  b'%PDF',
    '.jpg':  b'\xff\xd8\xff',
    '.jpeg': b'\xff\xd8\xff',
    '.png':  b'\x89PNG',
}

def validate_attachment(attachment: discord.Attachment) -> tuple[bool, str]:
    """Layer 1: Check basic rules before wasting bandwidth downloading."""
    if attachment.size > MAX_FILE_SIZE:
        return False, f"❌ File too large ({attachment.size // 1024 // 1024}MB). Maximum is 10MB."
    
    ext = os.path.splitext(attachment.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"❌ Unsupported format `{ext}`. Please send PDF, JPG, or PNG files only."
        
    return True, ""

def verify_file_integrity(file_path: str) -> bool:
    """Layer 2: Ensure the file bytes match its claimed extension."""
    ext = os.path.splitext(file_path)[1].lower()
    expected_magic = MAGIC_BYTES.get(ext)
    
    if not expected_magic:
        return False

    with open(file_path, 'rb') as f:
        header = f.read(len(expected_magic))
        
    return header == expected_magic

def sanitize_for_govt_portal(file_path: str) -> str:
    """Layer 3: Strip EXIF, convert PNG->JPG, and compress to 20-100KB."""
    ext = os.path.splitext(file_path)[1].lower()
    
    # We don't touch PDFs here, only images
    if ext == '.pdf':
        return file_path
        
    img = Image.open(file_path)
    
    # 1. Convert to pure RGB (strip alpha channel, CMYK, Palette, etc.)
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    # 2. Create a brand new image to permanently destroy EXIF metadata
    clean_img = Image.new('RGB', img.size)
    clean_img.putdata(list(img.getdata()))
    
    # 3. Binary Search Compression to hit the 20-100KB sweet spot
    lo, hi = 5, 95
    best_data = None
    
    while lo <= hi:
        mid = (lo + hi) // 2
        buffer = io.BytesIO()
        clean_img.save(buffer, format="JPEG", quality=mid)
        size_kb = buffer.tell() / 1024
        
        if size_kb < 20:
            lo = mid + 1
        elif size_kb > 100:
            hi = mid - 1
        else:
            best_data = buffer.getvalue()
            break
            
    # Fallback if we couldn't perfectly hit the range
    if best_data is None:
        best_data = buffer.getvalue()
        
    # 4. Save as a fresh .jpg and delete the old file if it was a .png
    new_file_path = os.path.splitext(file_path)[0] + ".jpg"
    with open(new_file_path, 'wb') as f:
        f.write(best_data)
        
    if file_path != new_file_path and os.path.exists(file_path):
        os.remove(file_path)
        
    return new_file_path

async def download_and_process(attachment: discord.Attachment, user_id: str, sanitize_for_portal: bool = False) -> tuple[bool, str]:
    """The Main Orchestrator: Downloads, verifies, and optionally sanitizes for govt portal upload.
    
    By default (sanitize_for_portal=False), keeps the crisp high-res original image for AI Vision extraction.
    When sanitize_for_portal=True, strips EXIF and compresses to 20-100KB JPG for strict govt portals.
    """
    # Run Layer 1
    is_valid, error_msg = validate_attachment(attachment)
    if not is_valid:
        return False, error_msg
        
    ext = os.path.splitext(attachment.filename)[1].lower()
    # Create a unique timestamp so files never overwrite each other
    timestamp = int(time.time())
    temp_file_path = os.path.join(TEMP_DIR, f"{user_id}_{timestamp}_raw{ext}")
    
    try:
        # Actually download the file from Discord
        await attachment.save(temp_file_path)
        
        # Run Layer 2
        if not verify_file_integrity(temp_file_path):
            os.remove(temp_file_path)
            return False, "❌ File integrity check failed. The file appears to be corrupted or disguised."
            
        # Run Layer 3 ONLY if explicitly requested for govt portal upload
        if sanitize_for_portal:
            final_path = sanitize_for_govt_portal(temp_file_path)
            return True, final_path
            
        return True, temp_file_path
        
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return False, f"❌ Failed to process document: {str(e)}"