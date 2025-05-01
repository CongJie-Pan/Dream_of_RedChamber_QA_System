"""
Text Ingestion Script for LightRAG

This script processes text files from a specified directory and imports them into a LightRAG 
knowledge database. It handles reading various file formats, splits content into manageable chunks,
and inserts the content into the database for later retrieval.

Features:
- Processes all text files in a specified directory
- Handles various file encodings automatically
- Splits large texts into semantically meaningful chunks
- Provides detailed progress reporting and error handling
- Generates comprehensive metadata about the import process
- Supports command-line arguments for customizing behavior

Usage:
    python insert_pydantic_docs.py [--working-dir DIR] [--data-dir DIR]

Author: CongJie Pan
Date: April 2025
"""

import os
import asyncio
import time
import sys
import psutil
import traceback
import glob
import logging
import json
from pathlib import Path
from datetime import datetime
from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
import dotenv
import re

# store the logs in the logs/lightrag_import directory
os.makedirs("logs/lightrag_import", exist_ok=True)

# Configure logging
# Set up both file and console logging with timestamps and log levels
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/lightrag_import/lightrag_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("lightrag_import")

# Load environment variables from .env file (containing API keys, etc.)
dotenv.load_dotenv()

# Default working directory for the LightRAG database
# This can be overridden via command-line arguments
DEFAULT_WORKING_DIR = "./basicSinogyAsk"

# Directory containing the text files to process
# This can be overridden via command-line arguments
DATA_DIR = "./data"

# Define chunk size for splitting content (in characters)
# Larger chunks provide more context but may exceed model token limits
CHUNK_SIZE = 100000  # 100K characters per chunk

def print_progress(message, success=True):
    """
    Print a formatted progress message and log it.
    
    This function provides consistent progress reporting by:
    1. Prefixing messages with a success/error indicator
    2. Printing to the console for immediate feedback
    3. Logging to the configured logger for permanent record
    4. Flushing stdout to ensure messages appear in real-time
    
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
    
    sys.stdout.flush()  # Ensure output is displayed immediately for real-time monitoring

def get_memory_usage():
    """
    Get current memory usage information of this process.
    
    This helps monitor resource usage during processing, which is useful for:
    - Debugging memory leaks
    - Ensuring the application stays within resource constraints
    - Performance optimization
    
    Returns:
        str: Formatted string with memory usage in MB
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return f"Memory: {memory_info.rss / (1024 * 1024):.2f} MB"

def find_text_files():
    """
    Find all text files in the data directory.
    
    This function:
    1. Checks if the data directory exists
    2. Finds all .txt files in the directory
    3. Reports detailed information about found files
    
    Returns:
        list: List of file paths (str) to process
    
    Raises:
        FileNotFoundError: If the data directory doesn't exist
    """
    print_progress(f"Searching for text files in: {DATA_DIR}")
    
    if not os.path.exists(DATA_DIR):
        error_msg = f"Data directory not found: {DATA_DIR}"
        print_progress(error_msg, False)
        raise FileNotFoundError(error_msg)
    
    # Get all text files in the data directory using glob pattern
    text_files = glob.glob(os.path.join(DATA_DIR, "*.txt"))
    
    if not text_files:
        warning_msg = f"No text files found in {DATA_DIR}"
        print_progress(warning_msg, False)
        logger.warning(warning_msg)
    else:
        # Format the list of files for better readability in logs
        file_list = "\n  - ".join([""] + [os.path.basename(f) for f in text_files])
        print_progress(f"Found {len(text_files)} text files:{file_list}")
    
    return text_files

def read_text_file(file_path) -> str:
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

def split_content_into_chunks(content, chunk_size=CHUNK_SIZE):
    """
    Split content into manageable chunks for processing.
    
    This function intelligently splits text by:
    1. Respecting paragraph boundaries where possible
    2. Falling back to line breaks if no paragraph breaks are found
    3. Ensuring chunks don't exceed the specified size
    
    This approach preserves semantic meaning better than naive character-based splitting.
    
    Args:
        content (str): The full text content to split
        chunk_size (int): Approximate maximum size of each chunk in characters
        
    Returns:
        list: List of text chunks (str)
    """
    # Try to split on paragraph boundaries near the chunk size
    chunks = []
    start = 0
    content_len = len(content)
    
    print_progress(f"Splitting content of {content_len} characters into chunks of ~{chunk_size} characters")
    
    while start < content_len:
        end = min(start + chunk_size, content_len)
        
        # If we're not at the end, try to find a natural breakpoint
        if end < content_len:
            # First priority: Look for double newline (paragraph break)
            paragraph_break = content.rfind('\n\n', start, end)
            if paragraph_break != -1:
                end = paragraph_break + 2  # Include the newlines
            else:
                # Second priority: Fall back to single newline if no paragraph break
                line_break = content.rfind('\n', start, end)
                if line_break != -1:
                    end = line_break + 1  # Include the newline
        
        chunks.append(content[start:end])
        start = end
    
    print_progress(f"Content split into {len(chunks)} chunks")
    return chunks

async def initialize_rag(working_dir=DEFAULT_WORKING_DIR):
    """
    Initialize the LightRAG instance with appropriate storage.
    
    This function:
    1. Creates the LightRAG instance with specified embedding and LLM functions
    2. Initializes storage backends
    3. Sets up the pipeline status for tracking processing
    
    Args:
        working_dir (str): The working directory for LightRAG storage
    
    Returns:
        LightRAG: Initialized LightRAG instance
    
    Raises:
        Exception: If initialization fails
    """
    print_progress(f"Creating LightRAG instance in {working_dir}")
    
    # Ensure working directory exists
    if not os.path.exists(working_dir):
        print_progress(f"Creating working directory: {working_dir}")
        os.makedirs(working_dir, exist_ok=True)
    else:
        print_progress(f"Using existing directory: {working_dir}")
    
    try:
        # Create LightRAG instance with OpenAI embedding and LLM functions
        rag = LightRAG(
            working_dir=working_dir,
            embedding_func=openai_embed,
            llm_model_func=gpt_4o_mini_complete
        )
        
        # Initialize storage backends (vector stores, document stores, etc.)
        print_progress("Initializing storages")
        start_time = time.time()
        await rag.initialize_storages()
        print_progress(f"Storages initialized (took {time.time() - start_time:.2f} seconds)")
        
        # Initialize pipeline status for tracking
        print_progress("Initializing pipeline status")
        await initialize_pipeline_status()
        print_progress("Pipeline status initialized")
        
        return rag
    except Exception as e:
        # Provide detailed error information for debugging
        error_details = traceback.format_exc()
        print_progress(f"Error initializing RAG: {e}", False)
        logger.error(f"Error details: {error_details}")
        raise

def save_metadata(working_dir, metadata):
    """
    Save metadata about the imported files to a JSON file.
    
    This metadata is useful for:
    - Tracking what has been imported
    - Diagnosing import issues
    - Providing statistics to the UI
    
    Args:
        working_dir (str): Working directory where to save the metadata
        metadata (dict): Dictionary with import metadata
    """
    metadata_file = os.path.join(working_dir, "import_metadata.json")
    try:
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print_progress(f"Metadata saved to {metadata_file}")
    except Exception as e:
        print_progress(f"Failed to save metadata: {e}", False)

async def process_text_file(rag, file_path, metadata):
    """
    Process a single text file and insert it into RAG.
    
    This function:
    1. Reads the text file
    2. Splits it into manageable chunks
    3. Inserts each chunk into the RAG system
    4. Reports progress and errors
    5. Updates metadata with file processing statistics
    
    Args:
        rag (LightRAG): Initialized LightRAG instance
        file_path (str): Path to the text file
        metadata (dict): Dictionary to update with file processing metadata
    
    Returns:
        bool: Success or failure of file processing
    """
    file_name = os.path.basename(file_path)
    print_progress(f"=== Processing file: {file_name} ===")
    
    try:
        # Read the text content
        content = read_text_file(file_path)
        
        # Split content into chunks
        chunks = split_content_into_chunks(content)
        total_chunks = len(chunks)
        
        # Prepare metadata for this file
        file_metadata = {
            "file_name": file_name,
            "file_path": file_path,
            "file_size_bytes": os.path.getsize(file_path),
            "character_count": len(content),
            "chunk_count": total_chunks,
            "start_time": datetime.now().isoformat(),
            "chunks_processed": 0,
            "errors": []
        }
        
        # Process each chunk
        start_time = time.time()
        for i, chunk in enumerate(chunks):
            chunk_start_time = time.time()
            print_progress(f"Processing chunk {i+1}/{total_chunks} ({len(chunk)} characters)...")
            
            try:
                # IMPORTANT: Use the async insert method instead of the synchronous one
                # This prevents the "event loop already running" error by using the existing event loop
                await rag.ainsert(chunk)
                
                chunk_time = time.time() - chunk_start_time
                memory_usage = get_memory_usage()
                print_progress(f"Chunk {i+1} processed in {chunk_time:.2f} seconds. {memory_usage}")
                
                # Update file metadata
                file_metadata["chunks_processed"] += 1
                
                # Report overall progress
                elapsed = time.time() - start_time
                progress = (i + 1) / total_chunks * 100
                est_remaining = elapsed / (i + 1) * (total_chunks - i - 1) if i > 0 else "calculating..."
                if isinstance(est_remaining, float):
                    est_remaining = f"{est_remaining:.2f} seconds"
                print_progress(f"File progress: {progress:.1f}% complete. Elapsed: {elapsed:.2f}s. Estimated remaining: {est_remaining}")
            
            except Exception as e:
                # Log detailed error information
                error_details = traceback.format_exc()
                print_progress(f"Error processing chunk {i+1}: {str(e)}", False)
                logger.error(f"Error details: {error_details}")
                
                # Record error in file metadata
                file_metadata["errors"].append({
                    "chunk_index": i,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                
                print_progress(f"Continuing with next chunk...", False)
        
        # Update final metadata
        file_metadata["end_time"] = datetime.now().isoformat()
        file_metadata["processing_time_seconds"] = time.time() - start_time
        file_metadata["success"] = file_metadata["chunks_processed"] == total_chunks
        
        metadata["files"].append(file_metadata)
        
        processing_time = time.time() - start_time
        print_progress(f"File {file_name} processed in {processing_time:.2f} seconds ({file_metadata['chunks_processed']}/{total_chunks} chunks successful)")
        
        return file_metadata["success"]
    
    except Exception as e:
        # Handle file-level errors
        error_details = traceback.format_exc()
        print_progress(f"Fatal error processing file {file_name}: {str(e)}", False)
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
        metadata["files"].append(file_metadata)
        metadata["failed_files"] += 1
        
        return False

async def main(working_dir=DEFAULT_WORKING_DIR):
    """
    Main function to process all text files and insert into RAG system.
    
    This function orchestrates the entire import process:
    1. Initializes metadata tracking
    2. Finds all text files to process
    3. Initializes the RAG system
    4. Processes each file and updates statistics
    5. Saves comprehensive metadata about the process
    
    Args:
        working_dir (str): Working directory for the RAG system
    """
    print_progress("=== Starting LightRAG Text Import Processing ===")
    logger.info(f"Starting import process with working directory: {working_dir}")
    
    # Initialize metadata dictionary to track the entire import process
    metadata = {
        "import_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "start_time": datetime.now().isoformat(),
        "data_directory": os.path.abspath(DATA_DIR),
        "working_directory": os.path.abspath(working_dir),
        "files": [],
        "total_files": 0,
        "successful_files": 0,
        "failed_files": 0
    }
    
    try:
        # Find all text files to process
        text_files = find_text_files()
        metadata["total_files"] = len(text_files)
        
        if not text_files:
            print_progress("No files to process, exiting.", False)
            metadata["end_time"] = datetime.now().isoformat()
            save_metadata(working_dir, metadata)
            return
        
        # Initialize RAG instance
        print_progress("Initializing RAG system...")
        rag = await initialize_rag(working_dir)
        
        # Process each file
        for i, file_path in enumerate(text_files):
            print_progress(f"Processing file {i+1}/{len(text_files)}: {os.path.basename(file_path)}")
            success = await process_text_file(rag, file_path, metadata)
            if success:
                metadata["successful_files"] += 1
            else:
                metadata["failed_files"] += 1
        
        # Save final metadata
        metadata["end_time"] = datetime.now().isoformat()
        save_metadata(working_dir, metadata)
        
        print_progress(f"=== Processing Complete: {metadata['successful_files']}/{metadata['total_files']} files successfully processed ===")
        
    except Exception as e:
        # Handle process-level errors
        error_details = traceback.format_exc()
        print_progress(f"Fatal error during processing: {str(e)}", False)
        logger.critical(f"Fatal error: {error_details}")
        
        # Save metadata even in case of failure
        metadata["end_time"] = datetime.now().isoformat()
        metadata["error"] = str(e)
        save_metadata(working_dir, metadata)
        
        print_progress("=== Processing Failed ===", False)
        sys.exit(1)


if __name__ == "__main__":
    # Parse command line arguments for customizing behavior
    import argparse
    parser = argparse.ArgumentParser(description="Import text files into LightRAG")
    parser.add_argument("--working-dir", "-w", default=DEFAULT_WORKING_DIR, 
                        help=f"Working directory for LightRAG (default: {DEFAULT_WORKING_DIR})")
    parser.add_argument("--data-dir", "-d", default=DATA_DIR,
                        help=f"Directory containing text files to process (default: {DATA_DIR})")
    
    args = parser.parse_args()
    
    # Update global variables based on args
    if args.data_dir != DATA_DIR:
        DATA_DIR = args.data_dir
        print_progress(f"Using custom data directory: {DATA_DIR}")
    
    # Run the main function with asyncio
    asyncio.run(main(args.working_dir))