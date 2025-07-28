# adobe-connect-dots
# Adobe Document Intelligence System

A sophisticated document intelligence system for the Adobe India Hackathon that performs PDF outline extraction and persona-driven document analysis.

## Features

### Round 1A: PDF Outline Extraction
- Extracts structured hierarchical outlines from PDF documents
- Identifies titles and headings (H1, H2, H3) with page numbers
- Supports documents up to 50 pages
- Fast processing (≤10 seconds per document)
- Outputs results in JSON format

### Round 1B: Persona-driven Document Intelligence  
- Analyzes 3-10 related PDFs based on specific persona and job requirements
- Ranks sections and subsections by relevance
- Supports diverse document types and personas
- Processes document collections in ≤60 seconds
- Provides detailed analysis with importance rankings

## Architecture

The system is built with a modular Python architecture:

- `main.py` - Main entry point and orchestration
- `src/outline_extractor.py` - PDF outline extraction (Round 1A)
- `src/persona_analyzer.py` - Persona-driven analysis (Round 1B)
- `src/pdf_processor.py` - Common PDF processing utilities
- `src/text_analyzer.py` - NLP and text analysis
- `src/utils.py` - Utility functions and helpers

## Technology Stack

- **Python 3.9+** - Core runtime
- **PyMuPDF (fitz)** - PDF processing and text extraction
- **spaCy** - Natural language processing
- **scikit-learn** - Machine learning and text analysis
- **Docker** - Containerization for consistent deployment

## Quick Start

### Using Docker (Recommended)

```bash
# Build the Docker image
docker build --platform linux/amd64 -t adobe-doc-intelligence:latest .

# Run Round 1A (outline extraction)
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  adobe-doc-intelligence:latest

# Run Round 1B (persona analysis)
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  -e PROCESSING_MODE=1b \
  -e PERSONA="PhD Researcher in Computer Science" \
  -e JOB_TO_BE_DONE="Literature review on machine learning" \
  --network none \
  adobe-doc-intelligence:latest
