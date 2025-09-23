# pdf_text_extractor.py
import fitz  # PyMuPDF
import pdfplumber
import os

class PDFTextExtractor:
    def __init__(self):
        self.extraction_method = None
        
    def extract_text_pymupdf(self, pdf_path):
        """Extract text using PyMuPDF (fastest method)"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text() + "\n"
            
            doc.close()
            self.extraction_method = "PyMuPDF"
            return text.strip()
            
        except Exception as e:
            print(f"PyMuPDF extraction failed: {e}")
            return None
    
    def extract_text_pdfplumber(self, pdf_path):
        """Extract text using pdfplumber (better for tables/complex layouts)"""
        try:
            text = ""
            
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            self.extraction_method = "pdfplumber"
            return text.strip()
            
        except Exception as e:
            print(f"pdfplumber extraction failed: {e}")
            return None
    
    def extract_text(self, pdf_path):
        """Main extraction method with fallback"""
        
        if not os.path.exists(pdf_path):
            return {"error": f"File not found: {pdf_path}"}
        
        if not pdf_path.lower().endswith('.pdf'):
            return {"error": "File must be a PDF"}
        
        # Try PyMuPDF first (fastest)
        text = self.extract_text_pymupdf(pdf_path)
        
        # If PyMuPDF fails or returns very little text, try pdfplumber
        if not text or len(text.strip()) < 50:
            print("PyMuPDF extraction insufficient, trying pdfplumber...")
            text = self.extract_text_pdfplumber(pdf_path)
        
        # Check if extraction was successful
        if not text or len(text.strip()) < 20:
            return {
                "error": "Could not extract sufficient text from PDF",
                "extracted_length": len(text) if text else 0,
                "suggestion": "PDF might be image-based (scanned) or corrupted"
            }
        
        return {
            "text": text,
            "method_used": self.extraction_method,
            "text_length": len(text),
            "status": "success"
        }
    
    def get_text_preview(self, text, max_chars=200):
        """Get a preview of extracted text"""
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."
