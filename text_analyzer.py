"""
Text Analysis utilities
Natural language processing and text analysis functionality
"""

import re
import logging
from typing import List, Dict, Any, Set, Tuple
from collections import Counter, defaultdict
import math

try:
    import spacy
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False

class TextAnalyzer:
    """Advanced text analysis and NLP functionality"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLP model if available
        self.nlp = None
        if NLP_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                self.logger.warning("spaCy English model not found")
        
        # Common stop words
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
            'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its',
            'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs'
        }
        
        # Academic and technical indicators
        self.academic_indicators = {
            'methodology', 'analysis', 'research', 'study', 'findings', 'results', 'conclusion',
            'hypothesis', 'experiment', 'data', 'statistical', 'significant', 'correlation',
            'regression', 'model', 'framework', 'theory', 'approach', 'method', 'technique'
        }
        
        self.technical_indicators = {
            'system', 'algorithm', 'implementation', 'architecture', 'design', 'specification',
            'protocol', 'interface', 'component', 'module', 'framework', 'platform', 'solution',
            'optimization', 'performance', 'scalability', 'efficiency', 'process', 'workflow'
        }
    
    def extract_keywords(self, text: str, max_keywords: int = 20) -> List[Tuple[str, float]]:
        """Extract important keywords from text with scores"""
        if not text:
            return []
        
        # Clean and tokenize text
        words = self._tokenize_and_clean(text)
        
        if NLP_AVAILABLE and self.nlp:
            return self._extract_keywords_nlp(text, max_keywords)
        else:
            return self._extract_keywords_basic(words, max_keywords)
    
    def _extract_keywords_nlp(self, text: str, max_keywords: int) -> List[Tuple[str, float]]:
        """Extract keywords using NLP"""
        try:
            doc = self.nlp(text[:1000000])  # Limit for memory
            
            # Extract noun phrases and important words
            keywords = defaultdict(float)
            
            # Noun phrases
            for chunk in doc.noun_chunks:
                if len(chunk.text) > 2 and chunk.text.lower() not in self.stop_words:
                    keywords[chunk.text.lower()] += 1.0
            
            # Named entities
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT']:
                    keywords[ent.text.lower()] += 2.0
            
            # Important tokens
            for token in doc:
                if (token.pos_ in ['NOUN', 'ADJ', 'VERB'] and 
                    not token.is_stop and 
                    not token.is_punct and 
                    len(token.text) > 2):
                    keywords[token.lemma_.lower()] += 0.5
            
            # Sort by score
            sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
            return sorted_keywords[:max_keywords]
            
        except Exception as e:
            self.logger.warning(f"NLP keyword extraction failed: {e}")
            words = self._tokenize_and_clean(text)
            return self._extract_keywords_basic(words, max_keywords)
    
    def _extract_keywords_basic(self, words: List[str], max_keywords: int) -> List[Tuple[str, float]]:
        """Extract keywords using basic frequency analysis"""
        # Count word frequencies
        word_freq = Counter(words)
        
        # Calculate TF-IDF like scores
        total_words = len(words)
        scored_words = []
        
        for word, freq in word_freq.items():
            # Simple scoring: frequency + length bonus + technical bonus
            score = freq / total_words
            
            # Length bonus for longer words
            if len(word) > 6:
                score *= 1.2
            
            # Technical/academic term bonus
            if word in self.academic_indicators or word in self.technical_indicators:
                score *= 1.5
            
            scored_words.append((word, score))
        
        # Sort by score
        scored_words.sort(key=lambda x: x[1], reverse=True)
        return scored_words[:max_keywords]
    
    def _tokenize_and_clean(self, text: str) -> List[str]:
        """Tokenize and clean text"""
        # Convert to lowercase and extract words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter stop words and short words
        filtered_words = [
            word for word in words 
            if word not in self.stop_words and len(word) > 2
        ]
        
        return filtered_words
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        if not text1 or not text2:
            return 0.0
        
        if NLP_AVAILABLE and self.nlp:
            try:
                doc1 = self.nlp(text1[:100000])
                doc2 = self.nlp(text2[:100000])
                
                if doc1.vector.any() and doc2.vector.any():
                    return doc1.similarity(doc2)
            except Exception:
                pass
        
        # Fallback to basic similarity
        return self._calculate_basic_similarity(text1, text2)
    
    def _calculate_basic_similarity(self, text1: str, text2: str) -> float:
        """Calculate basic similarity using word overlap"""
        words1 = set(self._tokenize_and_clean(text1))
        words2 = set(self._tokenize_and_clean(text2))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text"""
        if NLP_AVAILABLE and self.nlp:
            try:
                doc = self.nlp(text)
                return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
            except Exception:
                pass
        
        # Fallback to regex
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def classify_text_type(self, text: str) -> str:
        """Classify the type of text content"""
        text_lower = text.lower()
        
        # Count indicators
        academic_count = sum(1 for word in self.academic_indicators if word in text_lower)
        technical_count = sum(1 for word in self.technical_indicators if word in text_lower)
        
        # Look for specific patterns
        has_numbers = bool(re.search(r'\d+', text))
        has_citations = bool(re.search(r'\[(.*?)\]|\(.*?\d{4}.*?\)', text))
        has_formulas = bool(re.search(r'[=+\-*/]', text))
        
        # Classification logic
        if academic_count > 3 or has_citations:
            return "academic"
        elif technical_count > 3 or has_formulas:
            return "technical"
        elif has_numbers and any(word in text_lower for word in ['revenue', 'profit', 'financial']):
            return "financial"
        elif any(word in text_lower for word in ['chapter', 'section', 'definition', 'example']):
            return "educational"
        else:
            return "general"
    
    def extract_key_phrases(self, text: str, max_phrases: int = 10) -> List[str]:
        """Extract key phrases from text"""
        if NLP_AVAILABLE and self.nlp:
            try:
                doc = self.nlp(text[:100000])
                phrases = []
                
                # Noun phrases
                for chunk in doc.noun_chunks:
                    if len(chunk.text.split()) >= 2 and len(chunk.text) > 5:
                        phrases.append(chunk.text.strip())
                
                # Named entities
                for ent in doc.ents:
                    if len(ent.text) > 3:
                        phrases.append(ent.text.strip())
                
                # Remove duplicates and sort by length
                unique_phrases = list(set(phrases))
                unique_phrases.sort(key=len, reverse=True)
                
                return unique_phrases[:max_phrases]
                
            except Exception:
                pass
        
        # Fallback to regex patterns
        phrases = []
        
        # Find potential phrases (2-4 words)
        pattern = r'\b[A-Z][a-z]+(?: [A-Z][a-z]+){1,3}\b'
        matches = re.findall(pattern, text)
        phrases.extend(matches)
        
        # Find quoted phrases
        quoted = re.findall(r'"([^"]{5,50})"', text)
        phrases.extend(quoted)
        
        return list(set(phrases))[:max_phrases]
    
    def summarize_text(self, text: str, max_sentences: int = 3) -> str:
        """Generate a simple extractive summary"""
        if not text:
            return ""
        
        sentences = self.extract_sentences(text)
        if len(sentences) <= max_sentences:
            return " ".join(sentences)
        
        # Score sentences based on keyword frequency
        keywords = [kw for kw, _ in self.extract_keywords(text, 10)]
        
        sentence_scores = []
        for sentence in sentences:
            score = 0
            sentence_words = self._tokenize_and_clean(sentence)
            
            for word in sentence_words:
                if word in keywords:
                    score += 1
            
            # Normalize by sentence length
            if sentence_words:
                score = score / len(sentence_words)
            
            sentence_scores.append((sentence, score))
        
        # Sort by score and take top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in sentence_scores[:max_sentences]]
        
        return " ".join(top_sentences)
