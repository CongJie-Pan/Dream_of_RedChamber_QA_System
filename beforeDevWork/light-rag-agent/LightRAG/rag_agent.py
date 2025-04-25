"""
RAG Agent for Text-Based Question Answering

This module implements a Retrieval-Augmented Generation (RAG) agent that can answer questions
based on text data stored in a LightRAG knowledge base. The agent retrieves relevant
information from the knowledge base and generates answers using a language model.

Features:
- Uses LightRAG for efficient text retrieval
- Supports multiple retrieval strategies for robust performance
- Handles errors gracefully with fallback mechanisms
- Provides detailed logging for debugging and monitoring
- Supports both command-line and programmatic usage
- Responds in Chinese for Chinese questions, English for English questions

Usage:
    python rag_agent.py --question "Your question here" [--working-dir DIR]

Author: CongJie Pan
Date: April 2025
"""

import os
import sys
import argparse
from dataclasses import dataclass
import asyncio
import logging
import traceback
import json
from datetime import datetime

import dotenv
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
from openai import AsyncOpenAI

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status

# Configure logging with timestamps and appropriate formats
# Both file and console handlers are set up for comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"lightrag_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("lightrag_agent")

# Load environment variables from .env file (containing API keys, etc.)
dotenv.load_dotenv()

# Default working directory for the LightRAG database
# This can be overridden via command-line arguments
DEFAULT_WORKING_DIR = "./basicSinogyAsk"

def ensure_working_dir(working_dir):
    """
    Ensure the working directory exists, creating it if needed.
    
    This is important for proper LightRAG initialization, as it needs
    a directory to store its vector databases and other data.
    
    Args:
        working_dir (str): Path to the working directory
        
    Returns:
        str: The validated working directory path
    """
    if not os.path.exists(working_dir):
        logger.info(f"Creating working directory: {working_dir}")
        os.makedirs(working_dir, exist_ok=True)
    else:
        logger.info(f"Using existing directory: {working_dir}")
    return working_dir

# Check for OpenAI API key at module import time
# This prevents running the application without the required credentials
if not os.getenv("OPENAI_API_KEY"):
    logger.error("OPENAI_API_KEY environment variable not set.")
    logger.error("Please create a .env file with your OpenAI API key or set it in your environment.")
    sys.exit(1)


async def initialize_rag(working_dir=DEFAULT_WORKING_DIR):
    """
    Initialize the LightRAG instance for question answering.
    
    This function:
    1. Ensures the working directory exists
    2. Creates a LightRAG instance with appropriate embedding and LLM functions
    3. Initializes storage backends and pipeline status
    
    Args:
        working_dir (str): Working directory for LightRAG storage
        
    Returns:
        LightRAG: Initialized LightRAG instance
        
    Raises:
        Exception: If initialization fails for any reason
    """
    logger.info(f"Initializing LightRAG with working directory: {working_dir}")
    
    # Ensure working directory exists
    ensure_working_dir(working_dir)
    
    try:
        # Create LightRAG instance with OpenAI embedding and LLM functions
        rag = LightRAG(
            working_dir=working_dir,
            embedding_func=openai_embed,
            llm_model_func=gpt_4o_mini_complete
        )

        # Initialize storage backends (vector stores, document stores, etc.)
        logger.info("Initializing storages...")
        await rag.initialize_storages()
        logger.info("Storages initialized successfully")

        # Initialize pipeline status for tracking
        logger.info("Initializing pipeline status...")
        await initialize_pipeline_status()
        logger.info("Pipeline status initialized")

        return rag
    except Exception as e:
        # Provide detailed error information for debugging
        error_details = traceback.format_exc()
        logger.error(f"Error initializing RAG: {str(e)}")
        logger.error(f"Error details: {error_details}")
        raise


@dataclass
class RAGDeps:
    """
    Dependencies for the RAG agent.
    
    This class holds the LightRAG instance that the agent depends on
    for retrieving information. Using a dedicated class follows the
    dependency injection pattern, making the agent more testable and
    the dependencies more explicit.
    
    Attributes:
        lightrag (LightRAG): The LightRAG instance for retrievals
    """
    lightrag: LightRAG


# Create the Pydantic AI agent with a generic system prompt
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


async def safe_retrieve(lightrag, query, retries=2):
    """
    Safely retrieve information from LightRAG with error handling and retries.
    
    This function implements a robust retrieval strategy:
    1. Tries multiple query modes (mix, semantic, hybrid)
    2. Implements automatic retries with different parameters
    3. Provides fallback mechanisms for common error conditions
    4. Returns helpful error messages when retrieval fails
    
    Args:
        lightrag (LightRAG): The LightRAG instance
        query (str): Search query string
        retries (int): Number of retry attempts
        
    Returns:
        str: Retrieved text or error message
    """
    attempt = 0
    last_error = None
    
    while attempt <= retries:
        try:
            # Try to retrieve data with different query parameters based on attempt
            if attempt == 0:
                # First try with default mix mode
                logger.info(f"Retrieval attempt {attempt+1}: Using mix mode")
                result = await lightrag.aquery(query, param=QueryParam(mode="mix"))
            elif attempt == 1:
                # Second try with semantic mode only
                logger.info(f"Retrieval attempt {attempt+1}: Using semantic mode")
                result = await lightrag.aquery(query, param=QueryParam(mode="semantic"))
            else:
                # Last try with hybrid mode and more results
                logger.info(f"Retrieval attempt {attempt+1}: Using hybrid mode with expanded results")
                result = await lightrag.aquery(query, param=QueryParam(
                    mode="hybrid", 
                    max_chunks=10,
                    threshold=0.2
                ))
            
            # Count the number of document chunks to assess retrieval quality
            chunk_count = len(result.split('---')) if result else 0
            logger.info(f"Retrieved {chunk_count} document chunks")
            
            # If we got results, return them
            if result and chunk_count > 0:
                return result
            else:
                # No results but no error, try again with a different strategy
                logger.warning(f"No results returned for query: {query}, trying different retrieval strategy")
                attempt += 1
                continue
                
        except Exception as e:
            error_details = traceback.format_exc()
            last_error = str(e)
            logger.error(f"Error during retrieval attempt {attempt+1}: {last_error}")
            logger.error(f"Error details: {error_details}")
            
            # Special handling for 'file_path' error, which often indicates an issue with document metadata
            if "'file_path'" in last_error:
                logger.info("Detected 'file_path' error, trying direct storage access")
                try:
                    # Fallback: try to access the storage directly to get some content
                    # This bypasses the standard query mechanism when it's experiencing issues
                    chunks = await lightrag.storage_manager.get_storage("text_chunks").aget_all()
                    if chunks and len(chunks) > 0:
                        # Take up to 3 chunks that might be relevant based on simple keyword matching
                        sample_text = ""
                        count = 0
                        for chunk_id, chunk_data in chunks.items():
                            if count >= 3:
                                break
                            chunk_text = chunk_data.get('text', '')
                            # Very basic relevance check - containment of query terms
                            if any(term in chunk_text for term in query.split()):
                                sample_text += f"Document chunk {count+1}:\n{chunk_text}\n---\n"
                                count += 1
                        
                        if sample_text:
                            logger.info(f"Retrieved {count} chunks through direct storage access")
                            return sample_text
                except Exception as inner_e:
                    logger.error(f"Error in direct storage access fallback: {str(inner_e)}")
            
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
async def retrieve(context: RunContext[RAGDeps], search_query: str) -> str:
    """
    Retrieve relevant documents from vector database based on a search query.
    
    This is the main tool used by the agent to access the knowledge base.
    It's decorated with @agent.tool to make it available to the agent.
    
    Args:
        context (RunContext[RAGDeps]): The run context containing dependencies
        search_query (str): The search query to find relevant documents
        
    Returns:
        str: Formatted context information from the retrieved documents
    """
    logger.info(f"Retrieving information for query: {search_query}")
    
    # Use the safe retrieve function with retries and error handling
    result = await safe_retrieve(context.deps.lightrag, search_query)
    return result


async def run_rag_agent(question: str, working_dir=DEFAULT_WORKING_DIR) -> str:
    """
    Run the RAG agent to answer a question.
    
    This function:
    1. Initializes the LightRAG instance
    2. Sets up the agent dependencies
    3. Runs the agent with the question
    4. Returns the agent's response
    
    Args:
        question (str): The question to answer
        working_dir (str): Working directory for the LightRAG instance
        
    Returns:
        str: The agent's response or error message
    """
    logger.info(f"Running RAG agent with question: {question}")
    
    # Create dependencies and run the agent
    try:
        # Initialize LightRAG and set up dependencies
        lightrag = await initialize_rag(working_dir)
        deps = RAGDeps(lightrag=lightrag)
        
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


def main():
    """
    Main function to parse arguments and run the RAG agent from the command line.
    
    This function:
    1. Parses command-line arguments
    2. Runs the agent with the provided question
    3. Displays the response
    4. Handles errors gracefully
    """
    parser = argparse.ArgumentParser(description="Run a RAG agent with LightRAG")
    parser.add_argument("--question", "-q", required=True, help="The question to answer")
    parser.add_argument("--working-dir", "-w", default=DEFAULT_WORKING_DIR,
                      help=f"Working directory for LightRAG (default: {DEFAULT_WORKING_DIR})")
    
    args = parser.parse_args()
    
    # Display the question and run the agent
    print(f"Question: {args.question}")
    print("\nSearching for an answer...\n")
    
    try:
        # Run the agent asynchronously and get the response
        response = asyncio.run(run_rag_agent(args.question, args.working_dir))
        
        # Display the response
        print("\nResponse:")
        print(response)
    except Exception as e:
        # Handle errors gracefully
        print(f"\nError: {str(e)}")
        sys.exit(1)


# Run the main function when the script is executed directly
if __name__ == "__main__":
    main()
