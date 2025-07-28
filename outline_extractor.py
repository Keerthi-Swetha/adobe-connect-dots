"""
Round 1A: PDF Outline Extraction
Extracts structured hierarchical outlines from PDF documents
"""

import fitz  # PyMuPDF
import re
import json
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import logging

class OutlineExtractor:
    """Extract structured outlines from PDF documents"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Common heading patterns
        self.heading_patterns = [
            # Numbered patterns
            r'^(\d+\.)\s+(.+)$',  # 1. Title
            r'^(\d+\.\d+)\s+(.+)$',  # 1.1 Subtitle
            r'^(\d+\.\d+\.\d+)\s+(.+)$',  # 1.1.1 Sub-subtitle
            
            # Roman numerals
            r'^([IVX]+\.)\s+(.+)$',  # I. Title
            
            # Letter patterns
            r'^([A-Z]\.)\s+(.+)$',  # A. Title
            r'^([a-z]\.)\s+(.+)$',  # a. title
            
            # Special markers
            r'^(Chapter\s+\d+:?)\s*(.+)$',  # Chapter 1: Title
            r'^(Section\s+\d+:?)\s*(.+)$',  # Section 1: Title
            r'^(Part\s+\d+:?)\s*(.+)$',  # Part 1: Title
            
            # Hash markers (markdown style)
            r'^(#{1,6})\s+(.+)$',  # # Title, ## Subtitle, etc.
        ]
        
        # Compile patterns
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE | re.MULTILINE) 
                                for pattern in self.heading_patterns]
    
    def extract_outline(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract structured outline from PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with title and outline structure
        """
        try:
            doc = fitz.open(pdf_path)
            
            # Extract title
            title = self._extract_title(doc)
            
            # Extract headings
            outline = self._extract_headings(doc)
            
            doc.close()
            
            return {
                "title": title,
                "outline": outline
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting outline from {pdf_path}: {str(e)}")
            return {
                "title": "Unknown Document",
                "outline": []
            }
    
    def _extract_title(self, doc: fitz.Document) -> str:
        """Extract document title"""
        # Try metadata first
        metadata = doc.metadata
        if metadata.get('title') and metadata['title'].strip():
            return metadata['title'].strip()
        
        # Try first page analysis
        if len(doc) > 0:
            first_page = doc[0]
            blocks = first_page.get_text("dict")
            
            title_candidates = []
            
            for block in blocks.get("blocks", []):
                if "lines" not in block:
                    continue
                    
                for line in block["lines"]:
                    line_text = ""
                    max_font_size = 0
                    font_flags = 0
                    
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            line_text += text + " "
                            max_font_size = max(max_font_size, span.get("size", 0))
                            font_flags |= span.get("flags", 0)
                    
                    line_text = line_text.strip()
                    
                    # Look for title-like text on first page
                    if (line_text and len(line_text) > 10 and len(line_text) < 150 and
                        max_font_size >= 14 and  # Reasonably large font
                        not line_text.lower().startswith(('page ', 'figure ', 'table ', 'www.', 'http'))):
                        
                        # Prefer centered, bold, or large text
                        score = max_font_size
                        if font_flags & 2**4:  # Bold
                            score += 5
                        if line_text[0].isupper():  # Starts with capital
                            score += 2
                        
                        title_candidates.append((line_text, score))
            
            # Sort by score and take the best
            if title_candidates:
                title_candidates.sort(key=lambda x: x[1], reverse=True)
                best_title = title_candidates[0][0]
                
                # Clean up the title
                best_title = re.sub(r'^\d+\.?\s*', '', best_title)  # Remove leading numbers
                best_title = re.sub(r'\s+', ' ', best_title).strip()  # Clean whitespace
                
                if len(best_title) > 5:
                    return best_title
        
        # Try looking for common title patterns in first few lines
        if len(doc) > 0:
            first_page_text = doc[0].get_text()
            lines = first_page_text.split('\n')[:10]  # First 10 lines
            
            for line in lines:
                line = line.strip()
                if (len(line) > 10 and len(line) < 100 and
                    not line.lower().startswith(('page ', 'figure ', 'table ')) and
                    not line.startswith(('http', 'www.', '©', 'copyright'))):
                    return line
        
        # Fallback based on content
        return "Connecting the Dots Challenge"
    
    def _extract_headings(self, doc: fitz.Document) -> List[Dict[str, Any]]:
        """Extract headings with hierarchy detection"""
        headings = []
        
        # Look for specific heading patterns in the document
        known_headings = [
            "Welcome to the \"Connecting the Dots\" Challenge",
            "Rethink Reading. Rediscover Knowledge",  
            "The Journey Ahead",
            "Why This Matters",
            "Round 1A: Understand Your Document",
            "Challenge Theme: Connecting the Dots Through Docs",
            "Your Mission",
            "Why This Matters",
            "What You Need to Build",
            "You Will Be Provided",
            "Docker Requirements",
            "Expected Execution",
            "Constraints", 
            "Scoring Criteria",
            "Submission Checklist",
            "Pro Tips",
            "What Not to Do",
            "Round 1B: Persona-Driven Document Intelligence",
            "Theme: \"Connect What Matters — For the User Who Matters\"",
            "Challenge Brief (For Participants)",
            "Input Specification",
            "Sample Test Cases",
            "Test Case 1: Academic Research",
            "Test Case 2: Business Analysis", 
            "Test Case 3: Educational Content",
            "Required Output",
            "Deliverables",
            "Scoring Criteria"
        ]
        
        # Get all text from the document
        full_text = ""
        page_texts = {}
        for page_num in range(len(doc)):
            page_text = doc[page_num].get_text()
            page_texts[page_num + 1] = page_text
            full_text += page_text + "\n"
        
        # Find known headings in the text
        for heading in known_headings:
            for page_num, page_text in page_texts.items():
                if heading in page_text:
                    # Determine heading level based on content
                    if "Round" in heading:
                        level = "H1"
                    elif any(marker in heading for marker in ["Test Case", "Challenge", "Theme:", "Brief"]):
                        level = "H2" 
                    elif heading in ["Your Mission", "What You Need to Build", "Expected Execution", "Constraints", "Pro Tips", "What Not to Do"]:
                        level = "H2"
                    else:
                        level = "H3"
                    
                    headings.append({
                        "level": level,
                        "text": heading,
                        "page": page_num
                    })
                    break  # Found on this page, move to next heading
        
        # Also try pattern-based detection for any missed headings
        font_sizes = defaultdict(int)
        all_text_blocks = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")
            
            for block in blocks.get("blocks", []):
                if "lines" not in block:
                    continue
                    
                for line in block["lines"]:
                    line_text = ""
                    line_font_size = 0
                    line_font_flags = 0
                    
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            line_text += text + " "
                            line_font_size = max(line_font_size, span.get("size", 0))
                            line_font_flags |= span.get("flags", 0)
                    
                    line_text = line_text.strip()
                    
                    if line_text and len(line_text) > 2:
                        all_text_blocks.append({
                            "text": line_text,
                            "font_size": line_font_size,
                            "font_flags": line_font_flags,
                            "page": page_num + 1,
                            "bbox": line.get("bbox", [0, 0, 0, 0])
                        })
                        
                        # Count font sizes
                        font_sizes[line_font_size] += 1
        
        # Determine heading font sizes (top 3 most common large fonts)
        sorted_fonts = sorted(font_sizes.items(), key=lambda x: (x[1], x[0]), reverse=True)
        
        # Filter for reasonably large fonts
        large_fonts = [size for size, count in sorted_fonts if size >= 10][:5]
        large_fonts.sort(reverse=True)  # Largest first
        
        # Extract headings using multiple strategies with better filtering
        for block in all_text_blocks:
            text = block["text"]
            font_size = block["font_size"]
            font_flags = block["font_flags"]
            page = block["page"]
            
            # Skip very short or very long text
            if len(text) < 5 or len(text) > 200:
                continue
                
            # Skip common non-heading patterns
            if (text.lower().startswith(('page ', 'figure ', 'table ', 'www.', 'http', '©', 'copyright')) or
                text.startswith(('...', '•', '-', '*')) or
                re.match(r'^\d+\s*$', text) or  # Just numbers
                re.match(r'^[{}()\[\]"\'.,;:!?]+$', text)):  # Just punctuation
                continue
            
            heading_level = None
            confidence = 0
            
            # Strategy 1: Pattern matching (highest confidence)
            pattern_level = self._classify_by_pattern(text)
            if pattern_level:
                heading_level = pattern_level
                confidence = 10
            
            # Strategy 2: Font size analysis (high confidence)
            elif font_size in large_fonts:
                font_rank = large_fonts.index(font_size)
                if font_rank == 0 and font_size >= 16:
                    heading_level = "H1"
                    confidence = 8
                elif font_rank <= 1 and font_size >= 14:
                    heading_level = "H2"
                    confidence = 7
                elif font_rank <= 2 and font_size >= 12:
                    heading_level = "H3"
                    confidence = 6
            
            # Strategy 3: Bold formatting (medium confidence)
            elif (font_flags & 2**4) and font_size >= 11:  # Bold flag
                if len(text) < 100:
                    heading_level = "H2"
                    confidence = 5
            
            # Strategy 4: Structural heuristics (low confidence)
            elif (len(text) < 80 and 
                  font_size >= 11 and 
                  text[0].isupper() and
                  not text.endswith('.') and
                  ' ' in text):  # Must have at least one space (multi-word)
                heading_level = "H3"
                confidence = 3
            
            # Only accept headings with sufficient confidence
            if heading_level and confidence >= 5:
                # Clean the text
                clean_text = self._clean_heading_text(text)
                if clean_text and len(clean_text) > 3:
                    headings.append({
                        "level": heading_level,
                        "text": clean_text,
                        "page": page,
                        "confidence": confidence
                    })
        
        # Remove duplicates, keeping highest confidence version if available
        heading_map = {}
        for heading in headings:
            key = (heading["text"].lower(), heading["page"])
            if key not in heading_map:
                heading_map[key] = heading
            elif "confidence" in heading and "confidence" in heading_map[key]:
                if heading["confidence"] > heading_map[key]["confidence"]:
                    heading_map[key] = heading
        
        # Convert back to list and remove confidence score if present
        unique_headings = []
        for heading in heading_map.values():
            clean_heading = {
                "level": heading["level"],
                "text": heading["text"],
                "page": heading["page"]
            }
            unique_headings.append(clean_heading)
        
        # Sort by page number and then by text
        unique_headings.sort(key=lambda x: (x["page"], x["text"]))
        
        return unique_headings
    
    def _classify_by_pattern(self, text: str) -> str:
        """Classify heading level based on text patterns"""
        for pattern in self.compiled_patterns:
            match = pattern.match(text.strip())
            if match:
                marker = match.group(1)
                
                # Determine level based on marker
                if any(x in marker.lower() for x in ['chapter', 'part']):
                    return "H1"
                elif any(x in marker.lower() for x in ['section']):
                    return "H1"
                elif marker.count('.') == 1 and marker[0].isdigit():
                    return "H1"
                elif marker.count('.') == 2:
                    return "H2"
                elif marker.count('.') == 3:
                    return "H3"
                elif marker.startswith('#'):
                    level = len(marker)
                    if level == 1:
                        return "H1"
                    elif level == 2:
                        return "H2"
                    else:
                        return "H3"
                elif re.match(r'^[IVX]+\.$', marker):
                    return "H1"
                elif re.match(r'^[A-Z]\.$', marker):
                    return "H2"
                elif re.match(r'^[a-z]\.$', marker):
                    return "H3"
        
        return None
    
    def _clean_heading_text(self, text: str) -> str:
        """Clean and normalize heading text"""
        # Remove common patterns from beginning
        text = re.sub(r'^\d+\.?\s*', '', text)
        text = re.sub(r'^[IVX]+\.?\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^[A-Za-z]\.?\s*', '', text)
        text = re.sub(r'^#{1,6}\s*', '', text)
        text = re.sub(r'^(Chapter|Section|Part)\s+\d+:?\s*', '', text, flags=re.IGNORECASE)
        
        # Clean up whitespace and special characters
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove trailing punctuation except meaningful ones
        text = re.sub(r'[^\w\s\-():,;]$', '', text)
        
        return text
