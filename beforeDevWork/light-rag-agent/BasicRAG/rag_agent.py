"""
RAG Agent for Text-Based Question Answering

This module implements a Retrieval-Augmented Generation (RAG) agent that can answer questions
based on text data stored in a ChromaDB knowledge base. The agent retrieves relevant
information from the knowledge base and generates answers using a language model.

Features:
- Uses ChromaDB for efficient text retrieval
- Supports multiple retrieval strategies for robust performance
- Handles errors gracefully with fallback mechanisms
- Provides detailed logging for debugging and monitoring
- Supports both command-line and programmatic usage
- Responds in Chinese for Chinese questions, English for English questions

Usage:
    python rag_agent.py --question "Your question here" [--collection NAME] [--db-dir DIR]

Author: CongJie Pan (adapted for BasicRAG)
Date: April 2023
"""

import os
import sys
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import logging
import traceback
import json
from datetime import datetime
import re
import time

import dotenv
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
from openai import AsyncOpenAI

from utils import (
    get_chroma_client,
    get_or_create_collection,
    query_collection,
    format_results_as_context
)

# store the logs in the logs/rag_agent directory
os.makedirs("logs/rag_agent", exist_ok=True)

# Configure logging with timestamps and appropriate formats
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/rag_agent/rag_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rag_agent")

# Load environment variables for API keys
dotenv.load_dotenv()

# Default settings
DEFAULT_DB_DIR = "./chroma_db"
DEFAULT_COLLECTION = "document_store"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_N_RESULTS = 5

# Check for OpenAI API key at module import time
if not os.getenv("OPENAI_API_KEY"):
    logger.error("OPENAI_API_KEY environment variable not set.")
    logger.error("Please create a .env file with your OpenAI API key or set it in your environment.")
    sys.exit(1)


def ensure_db_dir(db_dir):
    """
    Ensure the database directory exists, creating it if needed.
    
    This is important for proper ChromaDB initialization, as it needs
    a directory to store its databases and other data.
    
    Args:
        db_dir (str): Path to the database directory
        
    Returns:
        str: The validated database directory path
    """
    if not os.path.exists(db_dir):
        logger.info(f"Creating database directory: {db_dir}")
        os.makedirs(db_dir, exist_ok=True)
    else:
        logger.info(f"Using existing directory: {db_dir}")
    return db_dir


def is_chinese_query(query: str) -> bool:
    """
    Detect if a query is primarily in Chinese.
    
    This function uses a simple heuristic to determine if a query
    contains Chinese characters, which helps in providing responses
    in the appropriate language.
    
    Args:
        query (str): The query text to check
        
    Returns:
        bool: True if the query contains Chinese characters, False otherwise
    """
    # Simple heuristic: if the query contains Chinese characters
    chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]')
    chinese_chars = chinese_char_pattern.findall(query)
    return len(chinese_chars) > 0


@dataclass
class RAGDeps:
    """
    Dependencies for the RAG agent.
    
    This class holds the ChromaDB dependencies that the agent depends on
    for retrieving information. Using a dedicated class follows the
    dependency injection pattern, making the agent more testable and
    the dependencies more explicit.
    
    Attributes:
        chroma_client (chromadb.PersistentClient): ChromaDB client instance
        collection_name (str): Name of the ChromaDB collection to use
        embedding_model (str): Name of the embedding model to use
    """
    chroma_client: Any  # chromadb.PersistentClient
    collection_name: str
    embedding_model: str


# Create the Pydantic AI agent with a system prompt
# The system prompt instructs the agent to use retrieval and handle
# both Chinese and English questions appropriately
agent = Agent(
    'openai:gpt-4o-mini',
    deps_type=RAGDeps,
    system_prompt="您是一個有幫助的助手，根據提供的文本來源回答問題。"
                  "使用檢索工具從文本數據庫中獲取相關信息後再回答。"
                  "如果文本中不包含答案，請明確說明當前文本中沒有該信息，並提供您最好的常識性回答。"
                  "當回答中文問題時，請用中文回答。對於英文問題，請用英文回答。"
)


async def safe_retrieve(collection, query_text, n_results=DEFAULT_N_RESULTS, retries=2):
    """
    Safely retrieve information from ChromaDB with error handling and retries.
    
    This function implements a robust retrieval strategy:
    1. Tries multiple query strategies
    2. Implements automatic retries with different parameters
    3. Provides fallback mechanisms for common error conditions
    4. Returns helpful error messages when retrieval fails
    
    Args:
        collection: ChromaDB collection
        query_text (str): Search query string
        n_results (int): Number of results to return
        retries (int): Number of retry attempts
        
    Returns:
        str: Retrieved text or error message
    """
    attempt = 0
    last_error = None
    
    while attempt <= retries:
        try:
            # Try to retrieve data with different parameters based on attempt
            if attempt == 0:
                # First try with default settings
                logger.info(f"Retrieval attempt {attempt+1}: Using default settings")
                results = query_collection(collection, query_text, n_results=n_results)
            elif attempt == 1:
                # Second try with more results
                logger.info(f"Retrieval attempt {attempt+1}: Increasing result count")
                results = query_collection(collection, query_text, n_results=n_results * 2)
            else:
                # Last try with even more results and different approach
                logger.info(f"Retrieval attempt {attempt+1}: Maximum retrieval")
                results = query_collection(collection, query_text, n_results=n_results * 3)
            
            # Format the results into a context string
            context = format_results_as_context(results)
            
            # Count the number of document chunks to assess retrieval quality
            result_count = len(results["documents"][0]) if results["documents"] else 0
            logger.info(f"Retrieved {result_count} document chunks")
            
            # If we got results, return them
            if result_count > 0:
                return context
            else:
                # No results but no error, try again with a different strategy
                logger.warning(f"No results returned for query: {query_text}, trying different retrieval strategy")
                attempt += 1
                continue
                
        except Exception as e:
            error_details = traceback.format_exc()
            last_error = str(e)
            logger.error(f"Error during retrieval attempt {attempt+1}: {last_error}")
            logger.error(f"Error details: {error_details}")
            
            # Increment attempt counter
            attempt += 1
    
    # If we've exhausted all retries, report the issue and return a helpful message
    error_msg = f"Failed to retrieve information after {retries+1} attempts. Last error: {last_error}"
    logger.error(error_msg)
    
    # Return a user-friendly message instead of just the error
    return ("I encountered a technical issue while searching for information on this topic. "
            "The system might not have enough data indexed on this subject yet. "
            f"Technical details: {last_error}")


@agent.tool
async def retrieve(context: RunContext[RAGDeps], search_query: str, n_results: int = DEFAULT_N_RESULTS) -> str:
    """
    Retrieve relevant documents from ChromaDB based on a search query.
    
    This is the main tool used by the agent to access the knowledge base.
    It's decorated with @agent.tool to make it available to the agent.
    
    Args:
        context (RunContext[RAGDeps]): The run context containing dependencies
        search_query (str): The search query to find relevant documents
        n_results (int): Number of results to return (default: 5)
        
    Returns:
        str: Formatted context information from the retrieved documents
    """
    logger.info(f"Retrieving information for query: {search_query}")
    
    # Get ChromaDB client and collection
    collection = get_or_create_collection(
        context.deps.chroma_client,
        context.deps.collection_name,
        embedding_model_name=context.deps.embedding_model
    )
    
    # Use the safe retrieve function with retries and error handling
    result = await safe_retrieve(collection, search_query, n_results)
    return result


async def run_rag_agent(
    question: str,
    collection_name: str = DEFAULT_COLLECTION,
    db_directory: str = DEFAULT_DB_DIR,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    n_results: int = DEFAULT_N_RESULTS
) -> str:
    """
    Run the RAG agent to answer a question.
    
    This function:
    1. Ensures the database directory exists
    2. Sets up the agent dependencies
    3. Runs the agent with the question
    4. Returns the agent's response
    
    Args:
        question (str): The question to answer
        collection_name (str): Name of the ChromaDB collection to use
        db_directory (str): Directory where ChromaDB data is stored
        embedding_model (str): Name of the embedding model to use
        n_results (int): Number of results to return from the retrieval
        
    Returns:
        str: The agent's response or error message
    """
    logger.info(f"Running RAG agent with question: {question}")
    
    # Create dependencies and run the agent
    try:
        # Ensure database directory exists
        ensure_db_dir(db_directory)
        
        # Check if the collection likely exists
        collection_metadata_path = os.path.join(db_directory, "chroma.sqlite3")
        if not os.path.exists(collection_metadata_path):
            logger.warning(f"ChromaDB database file not found at {collection_metadata_path}")
            logger.warning("Make sure you've run insert_docs.py to populate the database")
        
        # Set up ChromaDB client and dependencies
        client = get_chroma_client(db_directory)
        deps = RAGDeps(
            chroma_client=client,
            collection_name=collection_name,
            embedding_model=embedding_model
        )
        
        # Log if the query is detected as Chinese
        is_chinese = is_chinese_query(question)
        logger.info(f"Query language detected as: {'Chinese' if is_chinese else 'English'}")
        
        # Run the agent to answer the question
        logger.info("Executing agent query")
        result = await agent.run(question, deps=deps)
        logger.info("Agent query completed successfully")
        
        return result.data
    except Exception as e:
        # Handle errors and provide useful error messages
        error_details = traceback.format_exc()
        error_msg = f"Error running RAG agent: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Error details: {error_details}")
        return error_msg


def get_import_metadata(db_dir=DEFAULT_DB_DIR):
    """
    Get information about imported documents from metadata.
    
    This function reads the metadata file created during document import
    to provide statistics and information about the knowledge base.
    
    Args:
        db_dir (str): Database directory where metadata is stored
        
    Returns:
        dict: Information about the imported documents, or error message
    """
    try:
        metadata_file = os.path.join(db_dir, "import_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            logger.info(f"Successfully loaded metadata from {metadata_file}")
            return metadata
        else:
            logger.warning(f"No metadata file found at {metadata_file}")
            return {
                "error": "No import metadata found",
                "db_directory": os.path.abspath(db_dir),
                "exists": os.path.exists(db_dir)
            }
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error getting import metadata: {str(e)}")
        logger.error(f"Error details: {error_details}")
        return {
            "error": str(e),
            "db_directory": os.path.abspath(db_dir) if db_dir else "Not specified"
        }


def main():
    """
    Main function to parse arguments and run the RAG agent from the command line.
    
    This function:
    1. Parses command-line arguments
    2. Runs the agent with the provided question
    3. Displays the response
    4. Handles errors gracefully
    """
    parser = argparse.ArgumentParser(description="Run a RAG agent with ChromaDB")
    parser.add_argument("--question", "-q", required=True, help="The question to answer")
    parser.add_argument("--collection", "-c", default=DEFAULT_COLLECTION, 
                        help=f"Name of the ChromaDB collection (default: {DEFAULT_COLLECTION})")
    parser.add_argument("--db-dir", "-d", default=DEFAULT_DB_DIR, 
                        help=f"Directory where ChromaDB data is stored (default: {DEFAULT_DB_DIR})")
    parser.add_argument("--embedding-model", "-m", default=DEFAULT_EMBEDDING_MODEL, 
                        help=f"Name of the embedding model to use (default: {DEFAULT_EMBEDDING_MODEL})")
    parser.add_argument("--n-results", "-n", type=int, default=DEFAULT_N_RESULTS, 
                        help=f"Number of results to return from the retrieval (default: {DEFAULT_N_RESULTS})")
    
    args = parser.parse_args()
    
    # Display the question and run the agent
    print(f"Question: {args.question}")
    print("\nSearching for an answer...\n")
    
    try:
        # Run the agent asynchronously and get the response
        response = asyncio.run(run_rag_agent(
            args.question,
            collection_name=args.collection,
            db_directory=args.db_dir,
            embedding_model=args.embedding_model,
            n_results=args.n_results
        ))
        
        # Display the response
        print("\nResponse:")
        print(response)
    except Exception as e:
        # Handle errors gracefully
        print(f"\nError: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
