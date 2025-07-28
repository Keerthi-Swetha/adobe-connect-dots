"""
Round 1B: Persona-driven Document Intelligence
Analyzes documents based on specific persona and job-to-be-done requirements
"""

import fitz  # PyMuPDF
import json
import re
from typing import Dict, List, Any, Tuple
from datetime import datetime
import logging
from collections import defaultdict
import math

try:
    import spacy
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    logging.warning("Advanced NLP libraries not available. Using basic text analysis.")

class PersonaAnalyzer:
    """Analyze documents based on persona and job requirements"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLP models if available
        self.nlp = None
        if NLP_AVAILABLE:
            try:
                # Try to load a small English model
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                self.logger.warning("spaCy model not found. Using basic analysis.")
                self.nlp = None
        
        # Persona-specific keywords and priorities
        self.persona_keywords = {
            'researcher': ['methodology', 'analysis', 'results', 'conclusion', 'study', 'research', 'findings', 'data', 'experiment', 'hypothesis'],
            'student': ['definition', 'concept', 'example', 'theory', 'principle', 'formula', 'equation', 'problem', 'solution', 'summary'],
            'analyst': ['trend', 'performance', 'metric', 'indicator', 'comparison', 'analysis', 'revenue', 'growth', 'market', 'strategy'],
            'engineer': ['implementation', 'design', 'architecture', 'system', 'process', 'algorithm', 'specification', 'requirements', 'technical'],
            'manager': ['strategy', 'plan', 'objective', 'goal', 'process', 'workflow', 'team', 'resource', 'timeline', 'milestone'],
            'journalist': ['event', 'fact', 'quote', 'source', 'timeline', 'impact', 'significance', 'context', 'background', 'development']
        }
        
        # Job-specific keywords
        self.job_keywords = {
            'literature_review': ['methodology', 'related work', 'prior research', 'comparison', 'gap', 'contribution'],
            'exam_preparation': ['key concepts', 'important', 'definition', 'formula', 'example', 'practice', 'summary'],
            'financial_analysis': ['revenue', 'profit', 'loss', 'expenses', 'investment', 'return', 'performance', 'financial'],
            'technical_review': ['architecture', 'design', 'implementation', 'specification', 'performance', 'evaluation'],
            'market_analysis': ['market', 'competition', 'trend', 'opportunity', 'threat', 'positioning', 'share'],
            'content_summary': ['summary', 'overview', 'main points', 'key findings', 'conclusion', 'recommendation']
        }
    
    def analyze_documents(self, pdf_files: List[str], persona: str, job_to_be_done: str) -> Dict[str, Any]:
        """
        Analyze multiple documents based on persona and job requirements
        
        Args:
            pdf_files: List of PDF file paths
            persona: User persona description
            job_to_be_done: Specific task to accomplish
            
        Returns:
            Analysis result with ranked sections and subsections
        """
        try:
            # Extract content from all documents
            all_documents = []
            for pdf_file in pdf_files:
                doc_content = self._extract_document_content(pdf_file)
                if doc_content:
                    all_documents.append(doc_content)
            
            if not all_documents:
                return self._empty_result(pdf_files, persona, job_to_be_done)
            
            # Analyze persona and job requirements
            persona_keywords = self._extract_persona_keywords(persona)
            job_keywords = self._extract_job_keywords(job_to_be_done)
            
            # Extract and rank sections
            ranked_sections = self._extract_and_rank_sections(
                all_documents, persona_keywords, job_keywords, persona, job_to_be_done
            )
            
            # Extract and rank subsections
            ranked_subsections = self._extract_and_rank_subsections(
                all_documents, persona_keywords, job_keywords, ranked_sections
            )
            
            # Build result
            result = {
                "metadata": {
                    "input_documents": [doc["filename"] for doc in all_documents],
                    "persona": persona,
                    "job_to_be_done": job_to_be_done,
                    "processing_timestamp": datetime.now().isoformat()
                },
                "extracted_sections": ranked_sections[:20],  # Top 20 sections
                "subsection_analysis": ranked_subsections[:30]  # Top 30 subsections
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in document analysis: {str(e)}")
            return self._empty_result(pdf_files, persona, job_to_be_done)
    
    def _extract_document_content(self, pdf_path: str) -> Dict[str, Any]:
        """Extract structured content from a PDF document"""
        try:
            doc = fitz.open(pdf_path)
            filename = pdf_path.split('/')[-1]
            
            content = {
                "filename": filename,
                "pages": [],
                "sections": [],
                "all_text": ""
            }
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                
                content["pages"].append({
                    "page_number": page_num + 1,
                    "text": page_text
                })
                
                content["all_text"] += page_text + "\n"
                
                # Extract sections from this page
                sections = self._extract_sections_from_page(page, page_num + 1)
                content["sections"].extend(sections)
            
            doc.close()
            return content
            
        except Exception as e:
            self.logger.error(f"Error extracting content from {pdf_path}: {str(e)}")
            return None
    
    def _extract_sections_from_page(self, page: fitz.Page, page_number: int) -> List[Dict[str, Any]]:
        """Extract sections from a single page"""
        sections = []
        blocks = page.get_text("dict")
        
        current_section = None
        current_text = []
        
        for block in blocks.get("blocks", []):
            if "lines" not in block:
                continue
                
            for line in block["lines"]:
                line_text = ""
                is_heading = False
                font_size = 0
                
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if text:
                        line_text += text + " "
                        font_size = max(font_size, span.get("size", 0))
                        
                        # Check if this looks like a heading
                        if span.get("flags", 0) & 2**4 or font_size > 12:  # Bold or large
                            is_heading = True
                
                line_text = line_text.strip()
                
                if line_text:
                    if is_heading and len(line_text) < 200:
                        # Save previous section
                        if current_section and current_text:
                            current_section["content"] = "\n".join(current_text)
                            sections.append(current_section)
                        
                        # Start new section
                        current_section = {
                            "title": line_text,
                            "page": page_number,
                            "content": ""
                        }
                        current_text = []
                    else:
                        current_text.append(line_text)
        
        # Save last section
        if current_section and current_text:
            current_section["content"] = "\n".join(current_text)
            sections.append(current_section)
        
        return sections
    
    def _extract_persona_keywords(self, persona: str) -> List[str]:
        """Extract relevant keywords based on persona"""
        persona_lower = persona.lower()
        keywords = []
        
        # Match predefined persona types
        for persona_type, words in self.persona_keywords.items():
            if persona_type in persona_lower:
                keywords.extend(words)
        
        # Extract keywords from persona description
        words = re.findall(r'\b\w+\b', persona_lower)
        for word in words:
            if len(word) > 3 and word not in ['the', 'and', 'for', 'with', 'that', 'this']:
                keywords.append(word)
        
        return list(set(keywords))
    
    def _extract_job_keywords(self, job: str) -> List[str]:
        """Extract relevant keywords based on job-to-be-done"""
        job_lower = job.lower()
        keywords = []
        
        # Match predefined job types
        for job_type, words in self.job_keywords.items():
            if any(key_phrase in job_lower for key_phrase in job_type.split('_')):
                keywords.extend(words)
        
        # Extract keywords from job description
        words = re.findall(r'\b\w+\b', job_lower)
        for word in words:
            if len(word) > 3 and word not in ['the', 'and', 'for', 'with', 'that', 'this', 'need', 'want']:
                keywords.append(word)
        
        return list(set(keywords))
    
    def _extract_and_rank_sections(self, documents: List[Dict], persona_keywords: List[str], 
                                 job_keywords: List[str], persona: str, job: str) -> List[Dict[str, Any]]:
        """Extract and rank sections based on relevance"""
        all_sections = []
        
        for doc in documents:
            for section in doc["sections"]:
                if section["content"] and len(section["content"]) > 50:
                    # Calculate relevance score
                    score = self._calculate_relevance_score(
                        section["content"], section["title"], 
                        persona_keywords, job_keywords, persona, job
                    )
                    
                    all_sections.append({
                        "document": doc["filename"],
                        "page_number": section["page"],
                        "section_title": section["title"],
                        "importance_rank": score,
                        "content_preview": section["content"][:200] + "..." if len(section["content"]) > 200 else section["content"]
                    })
        
        # Sort by importance rank (descending)
        all_sections.sort(key=lambda x: x["importance_rank"], reverse=True)
        
        # Add rank numbers
        for i, section in enumerate(all_sections):
            section["importance_rank"] = i + 1
        
        return all_sections
    
    def _extract_and_rank_subsections(self, documents: List[Dict], persona_keywords: List[str], 
                                    job_keywords: List[str], top_sections: List[Dict]) -> List[Dict[str, Any]]:
        """Extract and rank subsections from top sections"""
        subsections = []
        
        # Get content from top sections
        for section in top_sections[:10]:  # Focus on top 10 sections
            doc_name = section["document"]
            page_num = section["page_number"]
            
            # Find the document and page
            for doc in documents:
                if doc["filename"] == doc_name:
                    for page in doc["pages"]:
                        if page["page_number"] == page_num:
                            # Split content into paragraphs
                            paragraphs = self._split_into_paragraphs(page["text"])
                            
                            for para in paragraphs:
                                if len(para) > 100:  # Meaningful paragraphs only
                                    score = self._calculate_relevance_score(
                                        para, "", persona_keywords, job_keywords, "", ""
                                    )
                                    
                                    subsections.append({
                                        "document": doc_name,
                                        "page_number": page_num,
                                        "refined_text": para[:300] + "..." if len(para) > 300 else para,
                                        "relevance_score": score
                                    })
        
        # Sort by relevance score
        subsections.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return subsections
    
    def _calculate_relevance_score(self, content: str, title: str, persona_keywords: List[str], 
                                 job_keywords: List[str], persona: str, job: str) -> float:
        """Calculate relevance score for content"""
        content_lower = content.lower()
        title_lower = title.lower()
        
        score = 0.0
        
        # Keyword matching
        for keyword in persona_keywords:
            if keyword in content_lower:
                score += 2.0
            if keyword in title_lower:
                score += 3.0
        
        for keyword in job_keywords:
            if keyword in content_lower:
                score += 2.5
            if keyword in title_lower:
                score += 4.0
        
        # Content quality indicators
        if len(content) > 200:
            score += 1.0
        
        # Numbers and data (important for analysts)
        if re.search(r'\d+', content):
            score += 0.5
        
        # References and citations (important for researchers)
        if re.search(r'\[(.*?)\]|\(.*?\d{4}.*?\)', content):
            score += 1.0
        
        # Technical terms (important for engineers)
        technical_words = ['system', 'method', 'process', 'algorithm', 'framework', 'model']
        for word in technical_words:
            if word in content_lower:
                score += 0.5
        
        # Use TF-IDF if available
        if NLP_AVAILABLE and self.nlp:
            try:
                # Simple semantic similarity using spaCy
                doc_nlp = self.nlp(content[:1000])  # Limit length for performance
                persona_nlp = self.nlp(persona)
                job_nlp = self.nlp(job)
                
                # Calculate similarity
                if doc_nlp.vector.any() and persona_nlp.vector.any():
                    persona_sim = doc_nlp.similarity(persona_nlp)
                    score += persona_sim * 5.0
                
                if doc_nlp.vector.any() and job_nlp.vector.any():
                    job_sim = doc_nlp.similarity(job_nlp)
                    score += job_sim * 6.0
                    
            except Exception:
                pass  # Fall back to keyword matching
        
        return round(score, 2)
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into meaningful paragraphs"""
        # Split by double newlines first
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Further split long paragraphs
        result = []
        for para in paragraphs:
            para = para.strip()
            if len(para) > 500:
                # Split by sentences
                sentences = re.split(r'[.!?]+', para)
                current_para = ""
                for sentence in sentences:
                    if len(current_para + sentence) > 500 and current_para:
                        result.append(current_para.strip())
                        current_para = sentence
                    else:
                        current_para += sentence + ". "
                if current_para.strip():
                    result.append(current_para.strip())
            elif len(para) > 50:
                result.append(para)
        
        return result
    
    def _empty_result(self, pdf_files: List[str], persona: str, job: str) -> Dict[str, Any]:
        """Return empty result structure"""
        return {
            "metadata": {
                "input_documents": [f.split('/')[-1] for f in pdf_files],
                "persona": persona,
                "job_to_be_done": job,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": [],
            "subsection_analysis": []
        }
