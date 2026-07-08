import os
import struct
import io
import discord
from PIL import Image
import time
import re
import glob

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

def get_user_dir(user_id: str) -> str:
    """Returns the dedicated user directory inside TEMP_DIR, creating it if needed."""
    user_dir = os.path.join(TEMP_DIR, str(user_id))
    os.makedirs(user_dir, exist_ok=True)
    return user_dir

def update_user_manifest(user_id: str, doc_type: str, file_path: str, status: str = "Verified & Saved", details: str = ""):
    """Maintains a clean user_manifest.md inside temp_documents/{user_id}/ for agent/Playwright reference."""
    try:
        user_dir = get_user_dir(user_id)
        manifest_path = os.path.join(user_dir, "user_manifest.md")
        timestamp_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        
        rel_path = os.path.basename(file_path)
        display_type = doc_type.replace('_', ' ').title()
        new_row = f"| **{display_type}** | `{rel_path}` | {timestamp_str} | {status} | {details} |\n"
        
        if not os.path.exists(manifest_path):
            header = (
                f"# Document Manifest for User `{user_id}`\n\n"
                "This summary lists all uploaded and processed document files for this user. "
                "Downstream agents (like Playwright automation) can reference this manifest for verification.\n\n"
                "| Document Type | File Name | Last Updated | Status | Details |\n"
                "|---|---|---|---|---|\n"
            )
            with open(manifest_path, "w", encoding="utf-8") as f:
                f.write(header + new_row)
        else:
            with open(manifest_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # If this doc_type already exists in table, replace the row, else append
            lines = content.splitlines()
            updated_lines = []
            row_found = False
            for line in lines:
                if line.startswith(f"| **{display_type}** |"):
                    updated_lines.append(new_row.strip())
                    row_found = True
                else:
                    updated_lines.append(line)
            if not row_found:
                updated_lines.append(new_row.strip())
                
            with open(manifest_path, "w", encoding="utf-8") as f:
                f.write("\n".join(updated_lines) + "\n")
    except Exception as e:
        print(f"[Manifest Update Error] Could not update user_manifest.md for {user_id}: {e}")

async def download_and_process(attachment: discord.Attachment, user_id: str, sanitize_for_portal: bool = False, doc_type_label: str = "unknown") -> tuple[bool, str]:
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
    clean_label = re.sub(r'[^a-zA-Z0-9_]', '_', doc_type_label.lower().strip())
    
    user_dir = get_user_dir(user_id)
    temp_file_path = os.path.join(user_dir, f"{clean_label}_{timestamp}_raw{ext}")
    
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
            update_user_manifest(user_id, doc_type_label, final_path, status="Sanitized for Portal", details="Compressed 20-100KB, EXIF stripped")
            return True, final_path
            
        update_user_manifest(user_id, doc_type_label, temp_file_path, status="Downloaded & Verified", details="High-res raw image kept for Vision LLM")
        return True, temp_file_path
        
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return False, f"❌ Failed to process document: {str(e)}"

def find_document_file(user_id: str, doc_type: str) -> str | None:
    """Finds the most recent file matching {doc_type}_* in temp_documents/{user_id}/.
    Falls back to checking flat directory {user_id}_{doc_type}_* for backward compatibility."""
    clean_label = re.sub(r'[^a-zA-Z0-9_]', '_', doc_type.lower().strip())
    user_dir = os.path.join(TEMP_DIR, str(user_id))
    
    # 1. Check user-specific directory first
    if os.path.exists(user_dir):
        pattern = os.path.join(user_dir, f"{clean_label}_*")
        matches = sorted(glob.glob(pattern), reverse=True)
        if matches:
            return matches[0]
            
    # 2. Fallback to flat folder pattern during transition/testing
    pattern_flat = os.path.join(TEMP_DIR, f"{user_id}_{clean_label}_*")
    matches_flat = sorted(glob.glob(pattern_flat), reverse=True)
    return matches_flat[0] if matches_flat else None

def get_portal_ready_file(user_id: str, doc_type: str) -> str | None:
    """Finds the raw file and returns a sanitized copy for govt portal upload.
    The original high-res stays intact for any future VLM re-processing."""
    raw_path = find_document_file(user_id, doc_type)
    if not raw_path:
        return None
    sanitized_path = sanitize_for_govt_portal(raw_path)
    update_user_manifest(user_id, doc_type, sanitized_path, status="Sanitized Copy Generated", details="Prepared for portal upload")
    return sanitized_path

def cleanup_user_files(user_id: str):
    """Deletes the entire user directory temp_documents/{user_id}/ after successful application."""
    import shutil
    user_dir = os.path.join(TEMP_DIR, str(user_id))
    if os.path.exists(user_dir):
        try:
            shutil.rmtree(user_dir)
            print(f"[Cleanup] Deleted document directory for user {user_id}")
        except Exception as e:
            print(f"[Cleanup Warning] Could not remove directory {user_dir}: {e}")
            
    # Also clean up any legacy flat files
    pattern_flat = os.path.join(TEMP_DIR, f"{user_id}_*")
    for file_path in glob.glob(pattern_flat):
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                pass