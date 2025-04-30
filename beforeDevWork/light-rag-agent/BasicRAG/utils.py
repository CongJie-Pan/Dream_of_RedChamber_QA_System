"""Utility functions for text processing and ChromaDB operations.

This module provides utilities for working with ChromaDB, including:
- Client and collection management
- Document addition and chunking
- Query and retrieval operations
- Result formatting

All functions include detailed error handling and type hints.
"""

import os
import pathlib
import logging
import traceback
import sys
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import chromadb
from chromadb.utils import embedding_functions
from more_itertools import batched

# Configure logging with timestamps and appropriate formats
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"chromadb_utils_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("chromadb_utils")


def get_chroma_client(persist_directory: str) -> chromadb.PersistentClient:
    """Get a ChromaDB client with the specified persistence directory.
    
    This function:
    - Creates the persistence directory if it doesn't exist
    - Establishes a connection to the ChromaDB store
    - Handles connection errors with detailed logging
    
    Args:
        persist_directory: Directory where ChromaDB will store its data
        
    Returns:
        A ChromaDB PersistentClient
        
    Raises:
        Exception: If client creation fails for any reason
    """
    try:
        # Create the directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        logger.info(f"Using ChromaDB directory at: {os.path.abspath(persist_directory)}")
        
        # Return the client
        client = chromadb.PersistentClient(persist_directory)
        logger.info("ChromaDB client created successfully")
        return client
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error creating ChromaDB client: {str(e)}")
        logger.error(f"Error details: {error_details}")
        raise


def get_or_create_collection(
    client: chromadb.PersistentClient,
    collection_name: str,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    distance_function: str = "cosine",
) -> chromadb.Collection:
    """Get an existing collection or create a new one if it doesn't exist.
    
    This function:
    - Creates embedding function with the specified model
    - Attempts to retrieve an existing collection
    - Creates a new collection if one doesn't exist
    - Handles exceptions with appropriate logging
    
    Args:
        client: ChromaDB client
        collection_name: Name of the collection
        embedding_model_name: Name of the embedding model to use
        distance_function: Distance function to use for similarity search
        
    Returns:
        A ChromaDB Collection
        
    Raises:
        Exception: If collection creation or retrieval fails
    """
    try:
        # Create embedding function
        logger.info(f"Using embedding model: {embedding_model_name}")
        embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        )
        
        # Try to get the collection, create it if it doesn't exist
        try:
            collection = client.get_collection(
                name=collection_name,
                embedding_function=embedding_func
            )
            logger.info(f"Retrieved existing collection: '{collection_name}'")
            return collection
        except chromadb.errors.InvalidCollectionException:
            logger.info(f"Collection '{collection_name}' does not exist, creating new collection")
            collection = client.create_collection(
                name=collection_name,
                embedding_function=embedding_func,
                metadata={"hnsw:space": distance_function}
            )
            logger.info(f"Created new collection: '{collection_name}'")
            return collection
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error with collection '{collection_name}': {str(e)}")
        logger.error(f"Error details: {error_details}")
        raise


def add_documents_to_collection(
    collection: chromadb.Collection,
    ids: List[str],
    documents: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    batch_size: int = 100,
) -> None:
    """Add documents to a ChromaDB collection in batches.
    
    This function:
    - Handles large document lists by processing in batches
    - Provides progress reporting during the import process
    - Creates default metadata when none is provided
    - Handles errors gracefully with detailed logging
    
    Args:
        collection: ChromaDB collection
        ids: List of document IDs
        documents: List of document texts
        metadatas: Optional list of metadata dictionaries for each document
        batch_size: Size of batches for adding documents
        
    Raises:
        Exception: If document addition fails
    """
    try:
        # Create default metadata if none provided
        if metadatas is None:
            logger.info("No metadata provided, using empty metadata for all documents")
            metadatas = [{}] * len(documents)
        
        # Document validation
        if len(ids) != len(documents) or len(ids) != len(metadatas):
            error_msg = f"Mismatch in lengths: ids ({len(ids)}), documents ({len(documents)}), metadatas ({len(metadatas) if metadatas else 0})"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Create document indices for batching
        document_indices = list(range(len(documents)))
        logger.info(f"Adding {len(documents)} documents in batches of {batch_size}")
        
        # Add documents in batches
        total_batches = (len(documents) + batch_size - 1) // batch_size
        for batch_num, batch in enumerate(batched(document_indices, batch_size)):
            # Get the start and end indices for the current batch
            start_idx = batch[0]
            end_idx = batch[-1] + 1  # +1 because end_idx is exclusive
            
            try:
                # Add the batch to the collection
                collection.add(
                    ids=ids[start_idx:end_idx],
                    documents=documents[start_idx:end_idx],
                    metadatas=metadatas[start_idx:end_idx],
                )
                
                # Log progress
                logger.info(f"Added batch {batch_num + 1}/{total_batches} ({start_idx}-{end_idx-1})")
            except Exception as e:
                # Log error but continue with next batch
                error_details = traceback.format_exc()
                logger.error(f"Error adding batch {batch_num + 1} (documents {start_idx}-{end_idx-1}): {str(e)}")
                logger.error(f"Error details: {error_details}")
                logger.info("Continuing with next batch...")
                
        logger.info(f"Successfully added {len(documents)} documents to collection '{collection.name}'")
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error adding documents to collection: {str(e)}")
        logger.error(f"Error details: {error_details}")
        raise


def query_collection(
    collection: chromadb.Collection,
    query_text: str,
    n_results: int = 5,
    where: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Query a ChromaDB collection for similar documents.
    
    This function:
    - Performs a semantic search across the collection
    - Retrieves the most relevant documents based on embedding similarity
    - Supports filtering with the 'where' parameter
    - Includes detailed logging for monitoring and debugging
    
    Args:
        collection: ChromaDB collection
        query_text: Text to search for
        n_results: Number of results to return
        where: Optional filter to apply to the query
        
    Returns:
        Query results containing documents, metadatas, distances, and ids
        
    Raises:
        Exception: If the query operation fails
    """
    try:
        logger.info(f"Querying collection '{collection.name}' with: '{query_text}'")
        logger.info(f"Requesting {n_results} results with filter: {where}")
        
        # Query the collection
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        # Log the number of results
        result_count = len(results["documents"][0]) if results["documents"] else 0
        logger.info(f"Query returned {result_count} results")
        
        return results
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error querying collection: {str(e)}")
        logger.error(f"Error details: {error_details}")
        raise


def format_results_as_context(query_results: Dict[str, Any]) -> str:
    """Format query results as a context string for the agent.
    
    This function:
    - Creates a standardized format for query results
    - Includes relevance scores and metadata in the output
    - Handles empty result sets gracefully
    - Creates a context string ready to be used by LLM agents
    
    Args:
        query_results: Results from a ChromaDB query
        
    Returns:
        Formatted context string with document content and metadata
    """
    try:
        # Handle empty results
        if not query_results or not query_results["documents"] or not query_results["documents"][0]:
            logger.warning("No results found in query_results")
            return "CONTEXT INFORMATION:\n\nNo relevant documents found."
        
        context = "CONTEXT INFORMATION:\n\n"
        
        for i, (doc, metadata, distance) in enumerate(zip(
            query_results["documents"][0],
            query_results["metadatas"][0],
            query_results["distances"][0]
        )):
            # Calculate relevance score (convert distance to similarity)
            relevance = 1 - distance
            
            # Add document information
            context += f"Document {i+1} (Relevance: {relevance:.2f}):\n"
            
            # Add metadata if available
            if metadata:
                for key, value in metadata.items():
                    context += f"{key}: {value}\n"
            
            # Add document content
            context += f"Content: {doc}\n\n"
        
        logger.info(f"Formatted {len(query_results['documents'][0])} documents as context")
        return context
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error formatting results as context: {str(e)}")
        logger.error(f"Error details: {error_details}")
        # Return a basic context with error information
        return f"CONTEXT INFORMATION:\n\nError formatting results: {str(e)}"


def split_text_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks for processing.
    
    This function:
    - Breaks long text into manageable chunks for embedding and storage
    - Maintains semantic context with overlapping chunks
    - Attempts to split at natural sentence or paragraph boundaries
    - Includes progress logging for large texts
    
    Args:
        text: The text to split
        chunk_size: Target size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
        
    Raises:
        ValueError: If invalid parameters are provided
    """
    # Validate parameters
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and less than chunk_size")
    
    if not text:
        logger.warning("Empty text provided to split_text_into_chunks")
        return []
    
    logger.info(f"Splitting text of {len(text)} characters into chunks (size={chunk_size}, overlap={overlap})")
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        # Calculate end position for this chunk
        end = min(start + chunk_size, text_length)
        
        # If we're not at the end of the text, try to find a good break point
        if end < text_length:
            # Look for paragraph break (double newline)
            paragraph_break = text.rfind('\n\n', start, end)
            if paragraph_break > start + (chunk_size // 2):
                end = paragraph_break + 2  # Include the newlines
            else:
                # Look for a single newline
                newline_pos = text.rfind('\n', start, end)
                if newline_pos > start + (chunk_size // 2):
                    end = newline_pos + 1  # Include the newline
                else:
                    # Last resort: Look for a space character
                    space_pos = text.rfind(' ', start, end)
                    if space_pos > start + (chunk_size // 2):
                        end = space_pos + 1  # Include the space
        
        # Add the chunk to our list
        chunks.append(text[start:end])
        
        # Move the start position for the next chunk, considering overlap
        start = max(start + 1, end - overlap)  # Ensure we always make progress
        
        # Log progress periodically
        if len(chunks) % 10 == 0:
            logger.info(f"Created {len(chunks)} chunks so far, processing {start}/{text_length} characters")
    
    logger.info(f"Text split into {len(chunks)} chunks")
    return chunks


def get_memory_usage() -> str:
    """Get current memory usage information of this process.
    
    This function is useful for monitoring resource usage during processing.
    
    Returns:
        str: Formatted string with memory usage in MB
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return f"Memory: {memory_info.rss / (1024 * 1024):.2f} MB"
    except ImportError:
        return "Memory usage unavailable (psutil not installed)"
    except Exception as e:
        return f"Error getting memory usage: {str(e)}"
