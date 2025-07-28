"""
PDF Processing utilities
Common functionality for PDF document handling
"""

import fitz  # PyMuPDF
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

class PDFProcessor:
    """Common PDF processing functionality"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_text_with_formatting(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from PDF with formatting information
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of pages with text and formatting data
        """
        try:
            doc = fitz.open(pdf_path)
            pages = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_data = {
                    "page_number": page_num + 1,
                    "text": page.get_text(),
                    "blocks": [],
                    "images": [],
                    "links": []
                }
                
                # Extract text blocks with formatting
                blocks = page.get_text("dict")
                for block in blocks.get("blocks", []):
                    if "lines" in block:
                        block_data = {
                            "type": "text",
                            "bbox": block.get("bbox", [0, 0, 0, 0]),
                            "lines": []
                        }
                        
                        for line in block["lines"]:
                            line_data = {
                                "bbox": line.get("bbox", [0, 0, 0, 0]),
                                "spans": []
                            }
                            
                            for span in line.get("spans", []):
                                span_data = {
                                    "text": span.get("text", ""),
                                    "font": span.get("font", ""),
                                    "size": span.get("size", 0),
                                    "flags": span.get("flags", 0),
                                    "color": span.get("color", 0),
                                    "bbox": span.get("bbox", [0, 0, 0, 0])
                                }
                                line_data["spans"].append(span_data)
                            
                            block_data["lines"].append(line_data)
                        
                        page_data["blocks"].append(block_data)
                    else:
                        # Image block
                        page_data["images"].append({
                            "bbox": block.get("bbox", [0, 0, 0, 0]),
                            "type": "image"
                        })
                
                # Extract links
                links = page.get_links()
                for link in links:
                    page_data["links"].append({
                        "bbox": link.get("from", [0, 0, 0, 0]),
                        "uri": link.get("uri", ""),
                        "page": link.get("page", -1)
                    })
                
                pages.append(page_data)
            
            doc.close()
            return pages
            
        except Exception as e:
            self.logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            return []
    
    def extract_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Extract PDF metadata"""
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata.copy()
            
            # Add document statistics
            metadata.update({
                "page_count": len(doc),
                "file_size": Path(pdf_path).stat().st_size,
                "encrypted": doc.is_encrypted,
                "pdf_version": doc.pdf_version()
            })
            
            doc.close()
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error extracting metadata from {pdf_path}: {str(e)}")
            return {}
    
    def extract_toc(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract table of contents if available"""
        try:
            doc = fitz.open(pdf_path)
            toc = doc.get_toc()
            
            toc_data = []
            for entry in toc:
                level, title, page = entry
                toc_data.append({
                    "level": level,
                    "title": title,
                    "page": page
                })
            
            doc.close()
            return toc_data
            
        except Exception as e:
            self.logger.error(f"Error extracting TOC from {pdf_path}: {str(e)}")
            return []
    
    def get_page_text(self, pdf_path: str, page_number: int) -> str:
        """Get text from specific page"""
        try:
            doc = fitz.open(pdf_path)
            
            if 0 <= page_number < len(doc):
                page = doc[page_number]
                text = page.get_text()
                doc.close()
                return text
            else:
                doc.close()
                return ""
                
        except Exception as e:
            self.logger.error(f"Error getting page {page_number} from {pdf_path}: {str(e)}")
            return ""
    
    def search_text(self, pdf_path: str, query: str) -> List[Dict[str, Any]]:
        """Search for text in PDF"""
        try:
            doc = fitz.open(pdf_path)
            results = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text_instances = page.search_for(query)
                
                for instance in text_instances:
                    results.append({
                        "page": page_num + 1,
                        "bbox": list(instance),
                        "text": query
                    })
            
            doc.close()
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching text in {pdf_path}: {str(e)}")
            return []
    
    def validate_pdf(self, pdf_path: str) -> bool:
        """Validate if file is a valid PDF"""
        try:
            doc = fitz.open(pdf_path)
            is_valid = len(doc) > 0
            doc.close()
            return is_valid
        except:
            return False
    
    def get_page_count(self, pdf_path: str) -> int:
        """Get number of pages in PDF"""
        try:
            doc = fitz.open(pdf_path)
            count = len(doc)
            doc.close()
            return count
        except:
            return 0
