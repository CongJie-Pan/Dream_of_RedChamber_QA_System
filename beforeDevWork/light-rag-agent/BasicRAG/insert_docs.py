"""
Text Ingestion Script for Basic RAG System

This script processes text files from a specified directory or URL and imports them into a ChromaDB
knowledge database. It handles reading various file formats, splits content into manageable chunks,
and inserts the content into the database for later retrieval.

Features:
- Processes text files in a specified directory or fetches from URLs
- Handles various file encodings automatically
- Splits large texts into semantically meaningful chunks
- Provides detailed progress reporting and error handling
- Generates comprehensive metadata about the import process
- Supports command-line arguments for customizing behavior

Usage:
    python insert_docs.py [--collection NAME] [--db-dir DIR] [--data-dir DIR]
                         [--url URL] [--chunk-size SIZE] [--overlap SIZE]
                         [--batch-size SIZE] [--embedding-model MODEL]

Author: CongJie Pan (adapted for BasicRAG)
Date: April 2023
"""

import os
import sys
import asyncio
import time
import psutil
import traceback
import glob
import logging
import json
import hashlib
import re
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import httpx

from utils import (
    get_chroma_client, 
    get_or_create_collection, 
    add_documents_to_collection,
    split_text_into_chunks,
    get_memory_usage
)
import dotenv

# store the logs in the logs/rag_import directory
os.makedirs("logs/rag_import", exist_ok=True)

# Configure logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/rag_import/rag_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rag_import")

# Load environment variables for API keys if needed
dotenv.load_dotenv()

# Default directories and settings
DEFAULT_DB_DIR = "./chroma_db"
DEFAULT_DATA_DIR = "./data"
DEFAULT_COLLECTION = "document_store"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 200
DEFAULT_BATCH_SIZE = 100

# Default URL for the Pydantic AI documentation (as example)
PYDANTIC_DOCS_URL = "https://ai.pydantic.dev/llms.txt"

def print_progress(message, success=True):
    """
    Print a formatted progress message and log it.
    
    This function provides consistent progress reporting by:
    1. Prefixing messages with a success/error indicator
    2. Printing to the console for immediate feedback
    3. Logging to the configured logger for permanent record
    
    Args:
        message (str): The progress message to display and log
        success (bool): Whether this is a success message (True) or error/warning (False)
    """
    prefix = "✅" if success else "❌"
    formatted_msg = f"{prefix} {message}"
    print(formatted_msg)
    
    # Also log to the logger with appropriate level
    if success:
        logger.info(message)
    else:
        logger.error(message)
    
    sys.stdout.flush()  # Ensure output is displayed immediately

def find_text_files(data_dir=DEFAULT_DATA_DIR):
    """
    Find all text files in the data directory.
    
    This function:
    1. Checks if the data directory exists
    2. Finds all .txt files in the directory
    3. Reports detailed information about found files
    
    Args:
        data_dir (str): Directory to search for text files
    
    Returns:
        list: List of file paths (str) to process
    
    Raises:
        FileNotFoundError: If the data directory doesn't exist
    """
    print_progress(f"Searching for text files in: {data_dir}")
    
    if not os.path.exists(data_dir):
        error_msg = f"Data directory not found: {data_dir}"
        print_progress(error_msg, False)
        raise FileNotFoundError(error_msg)
    
    # Get all text files in the data directory using glob pattern
    text_files = glob.glob(os.path.join(data_dir, "*.txt"))
    
    if not text_files:
        warning_msg = f"No text files found in {data_dir}"
        print_progress(warning_msg, False)
        logger.warning(warning_msg)
    else:
        # Format the list of files for better readability in logs
        file_list = "\n  - ".join([""] + [os.path.basename(f) for f in text_files])
        print_progress(f"Found {len(text_files)} text files:{file_list}")
    
    return text_files

def fetch_url_content(url: str) -> str:
    """
    Fetch content from a URL.
    
    This function:
    1. Makes an HTTP request to the specified URL
    2. Handles common HTTP errors with appropriate error messages
    3. Returns the text content if successful
    
    Args:
        url (str): URL to fetch content from
        
    Returns:
        str: Content from the URL
        
    Raises:
        Exception: If fetching content fails
    """
    print_progress(f"Fetching content from URL: {url}")
    
    try:
        response = httpx.get(url)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        content = response.text
        content_length = len(content)
        print_progress(f"Successfully fetched {content_length} characters from URL")
        
        return content
    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP error fetching URL: {e.response.status_code} - {e.response.reason_phrase}"
        print_progress(error_msg, False)
        raise Exception(error_msg)
    except httpx.RequestError as e:
        error_msg = f"Request error fetching URL: {str(e)}"
        print_progress(error_msg, False)
        raise Exception(error_msg)
    except Exception as e:
        error_msg = f"Error fetching URL {url}: {str(e)}"
        print_progress(error_msg, False)
        raise Exception(error_msg)

def read_text_file(file_path: str) -> str:
    """
    Read text from a file with encoding auto-detection.
    
    This function attempts to read a text file and handles various encodings:
    1. First tries UTF-8 (most common encoding)
    2. If that fails, attempts other common encodings (latin-1, gbk, big5, shift-jis)
    3. Reports detailed statistics about the file content
    
    Args:
        file_path (str): Path to the text file
    
    Returns:
        str: The content of the text file
    
    Raises:
        FileNotFoundError: If the file doesn't exist
        UnicodeDecodeError: If the file cannot be decoded with any attempted encoding
        Exception: For other reading errors
    """
    print_progress(f"Attempting to read text from: {file_path}")
    
    if not os.path.exists(file_path):
        error_msg = f"File not found: {file_path}"
        print_progress(error_msg, False)
        raise FileNotFoundError(error_msg)
    
    try:
        # First attempt with UTF-8 encoding (most common)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
        # Calculate and log basic text statistics
        char_count = len(content)
        line_count = content.count('\n') + 1
        print_progress(f"Successfully read {char_count} characters ({line_count} lines)")
        
        return content
    except UnicodeDecodeError as e:
        # If UTF-8 fails, try with different encodings commonly used for various languages
        encodings = ['latin-1', 'gbk', 'big5', 'shift-jis']
        for encoding in encodings:
            try:
                print_progress(f"Trying with {encoding} encoding...", False)
                with open(file_path, 'r', encoding=encoding) as file:
                    content = file.read()
                print_progress(f"Successfully read file with {encoding} encoding")
                return content
            except UnicodeDecodeError:
                continue
        
        # If all encoding attempts fail, report the error
        error_msg = f"Encoding error: {e}. Tried multiple encodings but none worked."
        print_progress(error_msg, False)
        raise
    except Exception as e:
        error_msg = f"Error reading file {file_path}: {e}"
        print_progress(error_msg, False)
        raise Exception(error_msg)

def extract_metadata(text: str, source: str) -> Dict[str, Any]:
    """
    Extract metadata from text content.
    
    This function:
    1. Analyzes the text to extract useful metadata
    2. Creates a consistent metadata structure for all documents
    3. Includes source information and content statistics
    
    Args:
        text (str): Text content to analyze
        source (str): Source identifier (file path or URL)
        
    Returns:
        dict: Metadata dictionary with content information
    """
    # Basic statistics
    metadata = {
        "source": source,
        "source_type": "url" if source.startswith("http") else "file",
        "char_count": len(text),
        "word_count": len(text.split()),
        "line_count": text.count('\n') + 1,
        "processing_time": datetime.now().isoformat()
    }
    
    # Try to extract title information (first line or first heading)
    lines = text.split('\n')
    if lines:
        first_line = lines[0].strip()
        if first_line:
            # Look for heading markers or just use the first line
            if first_line.startswith('#') and len(first_line) > 1:
                metadata["title"] = first_line.lstrip('#').strip()
            else:
                metadata["title"] = first_line
    
    # Generate a content hash for uniqueness checking
    metadata["content_hash"] = hashlib.md5(text.encode('utf-8')).hexdigest()
    
    return metadata

def process_content(
    content: str, 
    source: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE, 
    overlap: int = DEFAULT_OVERLAP
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """
    Process text content for insertion into the database.
    
    This function:
    1. Splits text into chunks of appropriate size
    2. Generates unique IDs for each chunk
    3. Creates detailed metadata for each chunk
    4. Returns lists ready for database insertion
    
    Args:
        content (str): Text content to process
        source (str): Source of the content (file path or URL)
        chunk_size (int): Size of each chunk in characters
        overlap (int): Number of characters to overlap between chunks
        
    Returns:
        Tuple containing lists of IDs, documents, and metadatas
    """
    print_progress(f"Processing content from {source}")
    print_progress(f"Splitting content into chunks (size: {chunk_size}, overlap: {overlap})")
    
    # Split content into chunks
    chunks = split_text_into_chunks(content, chunk_size, overlap)
    
    ids = []
    documents = []
    metadatas = []
    
    # Generate base metadata from full content
    base_metadata = extract_metadata(content, source)
    source_id = hashlib.md5(source.encode('utf-8')).hexdigest()[:8]
    
    print_progress(f"Processing {len(chunks)} chunks...")
    for i, chunk in enumerate(chunks):
        # Generate a unique ID for this chunk
        chunk_id = f"chunk-{source_id}-{i:04d}"
        
        # Create metadata for this chunk
        metadata = base_metadata.copy()
        metadata.update({
            "chunk_index": i,
            "chunk_count": len(chunks),
            "chunk_size": len(chunk),
            "chunk_id": chunk_id,
        })
        
        # Add to our lists
        ids.append(chunk_id)
        documents.append(chunk)
        metadatas.append(metadata)
        
        # Log progress periodically
        if i % 10 == 0 or i == len(chunks) - 1:
            print_progress(f"Processed chunk {i+1}/{len(chunks)}")
    
    return ids, documents, metadatas

def process_text_file(
    file_path: str,
    metadata_dict: Dict[str, Any],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP
) -> Tuple[List[str], List[str], List[Dict[str, Any]], bool]:
    """
    Process a single text file for database insertion.
    
    This function:
    1. Reads the text file
    2. Processes its content into chunks with metadata
    3. Updates the metadata dictionary with file processing statistics
    4. Returns data ready for database insertion
    
    Args:
        file_path (str): Path to the text file
        metadata_dict (dict): Dictionary to update with file processing metadata
        chunk_size (int): Size of each chunk in characters
        overlap (int): Number of characters to overlap between chunks
        
    Returns:
        Tuple containing:
            - List of document IDs
            - List of document chunks
            - List of metadata dictionaries
            - Success flag (bool)
    """
    file_name = os.path.basename(file_path)
    print_progress(f"=== Processing file: {file_name} ===")
    
    try:
        # Read the text content
        start_time = time.time()
        content = read_text_file(file_path)
        
        # Process the content into chunks with metadata
        ids, documents, metadatas = process_content(
            content, 
            file_path, 
            chunk_size=chunk_size, 
            overlap=overlap
        )
        
        # Prepare metadata for this file
        file_metadata = {
            "file_name": file_name,
            "file_path": file_path,
            "file_size_bytes": os.path.getsize(file_path),
            "character_count": len(content),
            "chunk_count": len(documents),
            "start_time": datetime.now().isoformat(),
            "processing_time_seconds": time.time() - start_time,
            "success": True
        }
        
        # Update metadata dictionary
        if "files" in metadata_dict:
            metadata_dict["files"].append(file_metadata)
        
        return ids, documents, metadatas, True
        
    except Exception as e:
        # Handle file-level errors
        error_details = traceback.format_exc()
        print_progress(f"Error processing file {file_name}: {str(e)}", False)
        logger.error(f"Error details: {error_details}")
        
        # Update metadata for failed file
        file_metadata = {
            "file_name": file_name,
            "file_path": file_path,
            "file_size_bytes": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
            "error": str(e),
            "success": False,
            "timestamp": datetime.now().isoformat()
        }
        
        # Update metadata dictionary
        if "files" in metadata_dict:
            metadata_dict["files"].append(file_metadata)
            metadata_dict["failed_files"] += 1
        
        return [], [], [], False

def process_url(
    url: str,
    metadata_dict: Dict[str, Any],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP
) -> Tuple[List[str], List[str], List[Dict[str, Any]], bool]:
    """
    Process content from a URL for database insertion.
    
    This function:
    1. Fetches content from the URL
    2. Processes the content into chunks with metadata
    3. Updates the metadata dictionary with URL processing statistics
    4. Returns data ready for database insertion
    
    Args:
        url (str): URL to fetch content from
        metadata_dict (dict): Dictionary to update with URL processing metadata
        chunk_size (int): Size of each chunk in characters
        overlap (int): Number of characters to overlap between chunks
        
    Returns:
        Tuple containing:
            - List of document IDs
            - List of document chunks
            - List of metadata dictionaries
            - Success flag (bool)
    """
    print_progress(f"=== Processing URL: {url} ===")
    
    try:
        # Fetch content from URL
        start_time = time.time()
        content = fetch_url_content(url)
        
        # Process the content into chunks with metadata
        ids, documents, metadatas = process_content(
            content, 
            url, 
            chunk_size=chunk_size, 
            overlap=overlap
        )
        
        # Prepare metadata for this URL
        url_metadata = {
            "url": url,
            "character_count": len(content),
            "chunk_count": len(documents),
            "start_time": datetime.now().isoformat(),
            "processing_time_seconds": time.time() - start_time,
            "success": True
        }
        
        # Update metadata dictionary
        if "urls" in metadata_dict:
            metadata_dict["urls"].append(url_metadata)
        
        return ids, documents, metadatas, True
        
    except Exception as e:
        # Handle URL-level errors
        error_details = traceback.format_exc()
        print_progress(f"Error processing URL {url}: {str(e)}", False)
        logger.error(f"Error details: {error_details}")
        
        # Update metadata for failed URL
        url_metadata = {
            "url": url,
            "error": str(e),
            "success": False,
            "timestamp": datetime.now().isoformat()
        }
        
        # Update metadata dictionary
        if "urls" in metadata_dict:
            metadata_dict["urls"].append(url_metadata)
            metadata_dict["failed_urls"] += 1
        
        return [], [], [], False

def save_metadata(db_dir: str, metadata: Dict[str, Any]) -> None:
    """
    Save metadata about the imported documents to a JSON file.
    
    This metadata is useful for:
    - Tracking what has been imported
    - Diagnosing import issues
    - Providing statistics to the UI
    
    Args:
        db_dir (str): Directory where to save the metadata
        metadata (dict): Dictionary with import metadata
    """
    metadata_file = os.path.join(db_dir, "import_metadata.json")
    try:
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print_progress(f"Metadata saved to {metadata_file}")
    except Exception as e:
        print_progress(f"Failed to save metadata: {e}", False)

def main():
    """
    Main function to process text files and URLs and insert them into ChromaDB.
    
    This function orchestrates the entire import process:
    1. Parses command-line arguments
    2. Initializes metadata tracking
    3. Finds all text files to process
    4. Creates the ChromaDB client and collection
    5. Processes each file/URL and inserts chunks into the database
    6. Saves comprehensive metadata about the process
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Import text files into ChromaDB")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, 
                        help=f"Name of the ChromaDB collection (default: {DEFAULT_COLLECTION})")
    parser.add_argument("--db-dir", default=DEFAULT_DB_DIR, 
                        help=f"Directory to store ChromaDB data (default: {DEFAULT_DB_DIR})")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help=f"Directory containing text files to process (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--url", 
                        help="URL to fetch content from instead of processing files")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, 
                        help=f"Name of the embedding model to use (default: {DEFAULT_EMBEDDING_MODEL})")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, 
                        help=f"Size of each text chunk (default: {DEFAULT_CHUNK_SIZE})")
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP, 
                        help=f"Overlap between chunks (default: {DEFAULT_OVERLAP})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, 
                        help=f"Batch size for adding documents (default: {DEFAULT_BATCH_SIZE})")
    
    args = parser.parse_args()
    
    print_progress("=== Starting Document Import Processing ===")
    logger.info(f"Starting import process with ChromaDB directory: {args.db_dir}")
    
    # Initialize metadata dictionary to track the entire import process
    metadata = {
        "import_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "start_time": datetime.now().isoformat(),
        "db_directory": os.path.abspath(args.db_dir),
        "collection_name": args.collection,
        "embedding_model": args.embedding_model,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "files": [],
        "urls": [],
        "total_files": 0,
        "successful_files": 0,
        "failed_files": 0,
        "total_urls": 0,
        "successful_urls": 0,
        "failed_urls": 0,
        "total_chunks": 0
    }
    
    try:
        # Determine the sources to process (URL or files)
        if args.url:
            # Process a single URL
            metadata["total_urls"] = 1
            print_progress(f"Processing URL: {args.url}")
            ids, documents, metadatas, success = process_url(
                args.url,
                metadata,
                chunk_size=args.chunk_size,
                overlap=args.overlap
            )
            if success:
                metadata["successful_urls"] += 1
            
            if not documents:
                print_progress("No content to process from URL, exiting.", False)
                metadata["end_time"] = datetime.now().isoformat()
                save_metadata(args.db_dir, metadata)
                return
        else:
            # Find all text files to process
            text_files = find_text_files(args.data_dir)
            metadata["total_files"] = len(text_files)
            
            if not text_files:
                print_progress("No files to process, exiting.", False)
                metadata["end_time"] = datetime.now().isoformat()
                save_metadata(args.db_dir, metadata)
                return
            
            # Process all files and combine the results
            all_ids = []
            all_documents = []
            all_metadatas = []
            
            for i, file_path in enumerate(text_files):
                print_progress(f"Processing file {i+1}/{len(text_files)}: {os.path.basename(file_path)}")
                ids, documents, metadatas, success = process_text_file(
                    file_path,
                    metadata,
                    chunk_size=args.chunk_size,
                    overlap=args.overlap
                )
                
                if success:
                    metadata["successful_files"] += 1
                    all_ids.extend(ids)
                    all_documents.extend(documents)
                    all_metadatas.extend(metadatas)
            
            # Update variables for ChromaDB insertion
            ids, documents, metadatas = all_ids, all_documents, all_metadatas
            metadata["total_chunks"] = len(documents)
            
            if not documents:
                print_progress("No content extracted from files, exiting.", False)
                metadata["end_time"] = datetime.now().isoformat()
                save_metadata(args.db_dir, metadata)
                return
        
        # Create ChromaDB client and collection
        print_progress(f"Connecting to ChromaDB at {args.db_dir}...")
        client = get_chroma_client(args.db_dir)
        collection = get_or_create_collection(
            client, 
            args.collection,
            embedding_model_name=args.embedding_model
        )
        
        # Add documents to the collection
        print_progress(f"Adding {len(documents)} chunks to collection '{args.collection}'...")
        add_documents_to_collection(
            collection,
            ids,
            documents,
            metadatas,
            batch_size=args.batch_size
        )
        
        # Update and save final metadata
        metadata["end_time"] = datetime.now().isoformat()
        metadata["total_chunks_added"] = len(documents)
        save_metadata(args.db_dir, metadata)
        
        print_progress(f"=== Import Complete: {len(documents)} chunks added to collection '{args.collection}' ===")
        
    except Exception as e:
        # Handle process-level errors
        error_details = traceback.format_exc()
        print_progress(f"Fatal error during processing: {str(e)}", False)
        logger.critical(f"Fatal error: {error_details}")
        
        # Save metadata even in case of failure
        metadata["end_time"] = datetime.now().isoformat()
        metadata["error"] = str(e)
        save_metadata(args.db_dir, metadata)
        
        print_progress("=== Processing Failed ===", False)
        sys.exit(1)


if __name__ == "__main__":
    main()
