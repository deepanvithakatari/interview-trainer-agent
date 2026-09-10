"""
Extract raw text from uploaded resume/JD files (PDF, DOCX, TXT).
"""
import io


def extract_text(uploaded_file):
    """
    uploaded_file: a Streamlit UploadedFile object.
    Returns plain text extracted from PDF, DOCX, or TXT.
    """
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    if name.endswith(".pdf"):
        import pdfplumber
        text = ""
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()

    elif name.endswith(".docx"):
        import docx
        doc = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs).strip()

    elif name.endswith(".txt"):
        return data.decode("utf-8", errors="replace").strip()

    else:
        raise ValueError(f"Unsupported file type: {name}. Please upload PDF, DOCX, or TXT.")
