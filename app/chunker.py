import re

def chunk_text(text: str, max_length: int = 1000) -> list:
    # Split by single newlines, then merge to max_length
    paras = [p.strip() for p in re.split(r'\n+', text) if p.strip()]
    chunks = []
    current = ""
    for para in paras:
        if len(current) + len(para) < max_length:
            current += " " + para
        else:
            if current:
                chunks.append(current.strip())
            current = para
    if current:
        chunks.append(current.strip())
    print(f"[DEBUG] Number of chunks: {len(chunks)}")
    if chunks:
        print(f"[DEBUG] First chunk (200 chars): {chunks[0][:200]}")
    return chunks