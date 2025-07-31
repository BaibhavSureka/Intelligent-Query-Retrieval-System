import requests
import pdfplumber
import docx
import tempfile
import os
import email
from email import policy
from email.parser import BytesParser

def download_file(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()
    # Enhanced file type detection
    url_lower = url.lower()
    if ".pdf" in url_lower:
        suffix = ".pdf"
    elif ".docx" in url_lower:
        suffix = ".docx"
    elif ".eml" in url_lower:
        suffix = ".eml"
    else:
        raise ValueError("Unsupported file type in URL. Only .pdf, .docx, and .eml are supported.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(response.content)
        return tmp.name

def load_pdf(path: str) -> str:
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

def load_docx(path: str) -> str:
    doc = docx.Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

def load_email(path: str) -> str:
    """Extract text content from email files (.eml)"""
    with open(path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)
    
    # Extract subject
    subject = msg.get('subject', '')
    
    # Extract body content
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body += part.get_content()
    else:
        body = msg.get_content()
    
    # Combine subject and body
    email_text = f"Subject: {subject}\n\n{body}"
    return email_text

def load_document(url: str) -> str:
    path = download_file(url)
    if path.endswith(".pdf"):
        text = load_pdf(path)
    elif path.endswith(".docx"):
        text = load_docx(path)
    elif path.endswith(".eml"):
        text = load_email(path)
    else:
        raise ValueError("Unsupported file type after download. Only .pdf, .docx, and .eml are supported.")
    os.remove(path)
    print("[DEBUG] Extracted text length:", len(text))
    print("[DEBUG] First 500 chars of extracted text:", text[:500])
    return text