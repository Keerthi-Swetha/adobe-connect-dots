"""
Utility functions for the document intelligence system
"""

import os
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import time

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    logging_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging_level)
    root_logger.addHandler(console_handler)
    
    return root_logger

def validate_pdf(pdf_path: str) -> bool:
    """Validate if file is a proper PDF"""
    logger = logging.getLogger(__name__)
    try:
        import fitz
        
        # Convert to string if it's a Path object
        pdf_path_str = str(pdf_path)
        
        logger.debug(f"Validating PDF: {pdf_path_str}")
        
        if not os.path.exists(pdf_path_str):
            logger.debug("File does not exist")
            return False
        
        # Check file extension
        if not pdf_path_str.lower().endswith('.pdf'):
            logger.debug("File extension check failed")
            return False
        
        # Check file size (should be > 0 and < 100MB for reasonable processing)
        file_size = os.path.getsize(pdf_path_str)
        if file_size == 0 or file_size > 100 * 1024 * 1024:
            logger.debug(f"File size check failed: {file_size} bytes")
            return False
        
        # Try to open with PyMuPDF
        doc = fitz.open(pdf_path_str)
        page_count = len(doc)
        is_valid = page_count > 0 and page_count <= 50  # Max 50 pages as per requirement
        doc.close()
        
        logger.debug(f"PDF validation successful: {page_count} pages")
        return is_valid
        
    except Exception as e:
        logger.error(f"PDF validation failed with exception: {str(e)}")
        return False

def load_json(file_path: str) -> Dict[str, Any]:
    """Load JSON file safely"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.getLogger(__name__).error(f"Error loading JSON from {file_path}: {e}")
        return {}

def save_json(data: Dict[str, Any], file_path: str) -> bool:
    """Save data to JSON file safely"""
    try:
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logging.getLogger(__name__).error(f"Error saving JSON to {file_path}: {e}")
        return False

def get_file_info(file_path: str) -> Dict[str, Any]:
    """Get basic file information"""
    try:
        stat = os.stat(file_path)
        return {
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "created": stat.st_ctime,
            "exists": True
        }
    except Exception:
        return {"exists": False}

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    import re
    text = re.sub(r'\s+', ' ', text)
    
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', '', text)
    
    return text.strip()

def extract_filename(path: str) -> str:
    """Extract filename without extension"""
    return Path(path).stem

def ensure_directory(dir_path: str) -> bool:
    """Ensure directory exists"""
    try:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logging.getLogger(__name__).error(f"Error creating directory {dir_path}: {e}")
        return False

def measure_time(func):
    """Decorator to measure function execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger = logging.getLogger(func.__module__)
        logger.info(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        
        return result
    return wrapper

def get_env_var(name: str, default: str = "", required: bool = False) -> str:
    """Get environment variable with optional default and validation"""
    value = os.getenv(name, default)
    
    if required and not value:
        raise ValueError(f"Required environment variable {name} is not set")
    
    return value

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def validate_json_schema(data: Dict[str, Any], schema_type: str) -> bool:
    """Validate JSON data against expected schema"""
    if schema_type == "outline":
        # Validate Round 1A output schema
        required_fields = ["title", "outline"]
        if not all(field in data for field in required_fields):
            return False
        
        if not isinstance(data["outline"], list):
            return False
        
        for item in data["outline"]:
            if not isinstance(item, dict):
                return False
            if not all(field in item for field in ["level", "text", "page"]):
                return False
            if item["level"] not in ["H1", "H2", "H3"]:
                return False
        
        return True
    
    elif schema_type == "persona_analysis":
        # Validate Round 1B output schema
        required_fields = ["metadata", "extracted_sections", "subsection_analysis"]
        if not all(field in data for field in required_fields):
            return False
        
        # Validate metadata
        metadata = data["metadata"]
        if not all(field in metadata for field in ["input_documents", "persona", "job_to_be_done", "processing_timestamp"]):
            return False
        
        return True
    
    return False

def create_sample_config() -> Dict[str, Any]:
    """Create sample configuration for testing"""
    return {
        "persona": "PhD Researcher in Computer Science",
        "job_to_be_done": "Prepare a literature review on machine learning algorithms for document analysis",
        "processing_mode": "1b",
        "max_sections": 20,
        "max_subsections": 30
    }

def log_system_info():
    """Log system information for debugging"""
    logger = logging.getLogger(__name__)
    
    logger.info("System Information:")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {sys.platform}")
    logger.info(f"Current working directory: {os.getcwd()}")
    
    # Check available memory (if psutil is available)
    try:
        import psutil
        memory = psutil.virtual_memory()
        logger.info(f"Available memory: {format_file_size(memory.available)}")
        logger.info(f"Total memory: {format_file_size(memory.total)}")
    except ImportError:
        logger.info("Memory information not available (psutil not installed)")
    
    # Check installed packages
    try:
        import pkg_resources
        installed_packages = [d.project_name for d in pkg_resources.working_set]
        logger.info(f"Key packages installed: {', '.join(sorted(installed_packages))}")
    except ImportError:
        logger.info("Package information not available")

class Timer:
    """Context manager for timing operations"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.logger = logging.getLogger(__name__)
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        duration = end_time - self.start_time
        self.logger.info(f"{self.name} completed in {duration:.2f} seconds")

def batch_process_files(input_dir: str, output_dir: str, processor_func, file_pattern: str = "*.pdf"):
    """Generic batch processing function"""
    logger = logging.getLogger(__name__)
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        logger.error(f"Input directory {input_dir} does not exist")
        return
    
    # Ensure output directory exists
    ensure_directory(output_dir)
    
    # Find files to process
    files = list(input_path.glob(file_pattern))
    
    if not files:
        logger.warning(f"No files matching {file_pattern} found in {input_dir}")
        return
    
    logger.info(f"Processing {len(files)} files...")
    
    for file_path in files:
        try:
            with Timer(f"Processing {file_path.name}"):
                result = processor_func(str(file_path))
                
                if result:
                    output_file = output_path / f"{file_path.stem}.json"
                    save_json(result, str(output_file))
                    logger.info(f"Saved result to {output_file}")
                
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
    
    logger.info("Batch processing completed")
