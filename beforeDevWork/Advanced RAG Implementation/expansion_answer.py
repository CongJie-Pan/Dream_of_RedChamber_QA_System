"""
# ================================================================================================
# Advanced RAG (Retrieval-Augmented Generation) System Implementation
# ================================================================================================
#
# This file implements an advanced RAG system with the following main features:
#
# 1. PDF document processing and text splitting
# 2. Vector database creation and querying
# 3. Implementation of query expansion techniques
# 4. Visualization of vector spaces using UMAP
#
# Key technical features:
# - Multi-level text splitting strategy
# - Integration of OpenAI API for query expansion
# - Using ChromaDB for vector storage and retrieval
# - Using UMAP for dimensionality reduction and visualization of high-dimensional vectors
#
# This system significantly improves retrieval accuracy and relevance through query expansion techniques,
# especially for complex queries.
# ================================================================================================
"""

import os
import logging
import matplotlib.pyplot as plt
import umap
import chromadb
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

# Import custom utility functions
from helper_utils import project_embeddings, word_wrap, extract_text_from_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for configuration
DEBUG = False  # Set to True to enable debug outputs

def load_environment():
    """
    Load environment variables from .env file
    
    Returns:
        OpenAI: Initialized OpenAI client
    """
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise EnvironmentError("OPENAI_API_KEY not found in environment variables")
    return OpenAI(api_key=openai_key)

def process_document(file_path, debug=DEBUG):
    """
    Read document and extract text from various file formats
    
    Args:
        file_path (str): Path to the document file
        debug (bool): Whether to print debug information
    
    Returns:
        list: List of text content from the document
    """
    logger.info(f"Processing document from: {file_path}")
    
    # Get file extension
    file_extension = os.path.splitext(file_path)[1].lower()
    
    # Extract text based on file type
    try:
        text, section_count = extract_text_from_file(file_path)
        
        # Split text into chunks by paragraphs
        pdf_texts = [t.strip() for t in text.split("\n\n") if t.strip()]
        
        # Filter empty strings
        pdf_texts = [text for text in pdf_texts if text]
        
        if debug:
            # Print the first 100 characters of the first text section
            print(f"Sample document content (first 100 chars):")
            print(word_wrap(pdf_texts[0], width=100))
            
        logger.info(f"Extracted {len(pdf_texts)} text sections from {file_extension} document with {section_count} pages/sections")
        return pdf_texts
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise

def split_text(pdf_texts, chunk_size=1000, token_size=256, debug=DEBUG):
    """
    Split text using two different methods for better chunking
    
    Args:
        pdf_texts (list): List of text content from PDF
        chunk_size (int): Size of chunks for character-based splitting
        token_size (int): Size of chunks for token-based splitting
        debug (bool): Whether to print debug information
    
    Returns:
        list: List of text chunks after splitting
    """
    logger.info("Starting text splitting process")
    
    # Step 1: Use recursive character splitter
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""], 
        chunk_size=chunk_size, 
        chunk_overlap=0
    )
    character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))
    
    if debug:
        print(f"\nAfter character splitting - Total chunks: {len(character_split_texts)}")
        print("Sample chunk (character split):")
        print(word_wrap(character_split_texts[min(10, len(character_split_texts)-1)]))
    
    # Step 2: Use SentenceTransformers-based token splitter
    token_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=0, 
        tokens_per_chunk=token_size
    )
    token_split_texts = []
    for text in character_split_texts:
        token_split_texts += token_splitter.split_text(text)
    
    if debug:
        print(f"\nAfter token splitting - Total chunks: {len(token_split_texts)}")
        print("Sample chunk (token split):")
        print(word_wrap(token_split_texts[min(10, len(token_split_texts)-1)]))
    
    logger.info(f"Text splitting complete. Generated {len(token_split_texts)} chunks")
    return token_split_texts

def create_vector_db(text_chunks, collection_name="microsoft-collection", embedding_function=None):
    """
    Create and populate a ChromaDB vector database with text chunks
    
    Args:
        text_chunks (list): List of text chunks to embed and store
        collection_name (str): Name for the ChromaDB collection
        embedding_function (object, optional): Custom embedding function to use. If None, a default one will be created.
    
    Returns:
        tuple: (ChromaDB collection, embedding function)
    """
    logger.info(f"Creating vector database with {len(text_chunks)} chunks")
    
    # Create embedding function instance if not provided
    if embedding_function is None:
        logger.info("No embedding function provided, using default English model")
        embedding_function = SentenceTransformerEmbeddingFunction()
    else:
        logger.info(f"Using provided embedding function")
    
    # Create a persistent ChromaDB client with a local directory
    # This ensures the database persists between application restarts
    persist_directory = os.path.join(os.path.dirname(__file__), "chroma_db")
    os.makedirs(persist_directory, exist_ok=True)
    
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    
    # Get or create collection
    try:
        # Try to get an existing collection
        chroma_collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
        logger.info(f"Using existing collection: {collection_name}")
    except Exception:
        # Create a new collection if it doesn't exist
        chroma_collection = chroma_client.create_collection(
            name=collection_name, 
            embedding_function=embedding_function
        )
        logger.info(f"Created new collection: {collection_name}")
        
        # Add text chunks to vector database
        ids = [str(i) for i in range(len(text_chunks))]
        chroma_collection.add(ids=ids, documents=text_chunks)
        logger.info(f"Added {len(text_chunks)} documents to collection")
    
    logger.info(f"Vector database ready with {chroma_collection.count()} records")
    return chroma_collection, embedding_function

def augment_query_generated(client, query, model="gpt-4o-mini"):
    """
    Use OpenAI's language model to generate example answers to query questions
    Supports multilingual queries including Chinese
    
    Args:
        client (OpenAI): OpenAI client
        query (str): The user's original query question
        model (str): The OpenAI model name used
        
    Returns:
        str: Example answer generated by AI
    """
    logger.info(f"Generating augmented query using model: {model}")
    
    # Detect if query is likely in Chinese or another non-English language
    has_chinese = any(u'\u4e00' <= char <= u'\u9fff' for char in query)
    
    # Set system prompt for AI assistant based on detected language
    if has_chinese:
        prompt = """你是一位专业研究助理，能够帮助用户分析文档。
        请针对用户的问题提供一个可能在文档中出现的示例答案。
        你的回答应该客观、准确，并且遵循中文表达习惯。回答时无需声明这是一个假设回答。"""
    else:
        prompt = """You are a helpful expert research assistant. 
        Provide an example answer to the given question, that might be found in a document.
        Your answer should be objective and factual without stating that this is a hypothetical answer."""
    
    # Build message list for OpenAI API
    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {
            "role": "user",
            "content": query
        },
    ]

    # Call OpenAI API to generate response
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    
    # Extract generated content from response
    content = response.choices[0].message.content
    logger.info("Query augmentation completed")
    return content

def perform_query(collection, query_text, n_results=5, is_augmented=False):
    """
    Perform vector retrieval on the database
    
    Args:
        collection: ChromaDB collection
        query_text (str): Query text to search for
        n_results (int): Number of results to return
        is_augmented (bool): Whether the query is augmented
        
    Returns:
        dict: Query results
    """
    query_type = "augmented" if is_augmented else "original"
    logger.info(f"Performing {query_type} query: '{query_text[:50]}...'")
    
    results = collection.query(
        query_texts=[query_text], 
        n_results=n_results,
        include=["documents", "embeddings"]
    )
    
    logger.info(f"Retrieved {len(results['documents'][0])} documents")
    return results

def visualize_embeddings(
    embeddings, 
    umap_transform, 
    original_query_embedding,
    augmented_query_embedding,
    retrieved_embeddings,
    original_query
):
    """
    Visualize the embeddings using UMAP projection
    
    Args:
        embeddings: All document embeddings
        umap_transform: Fitted UMAP transformer
        original_query_embedding: Embedding of original query
        augmented_query_embedding: Embedding of augmented query
        retrieved_embeddings: Embeddings of retrieved documents
        original_query: Original query text for plot title
    """
    logger.info("Generating visualization of embeddings")
    
    # Project embeddings to 2D using UMAP
    projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)
    projected_original_query_embedding = project_embeddings(
        original_query_embedding, umap_transform
    )
    projected_augmented_query_embedding = project_embeddings(
        augmented_query_embedding, umap_transform
    )
    projected_retrieved_embeddings = project_embeddings(
        retrieved_embeddings, umap_transform
    )

    # Create the visualization plot
    plt.figure(figsize=(10, 8))

    # Plot all document embeddings (gray dots)
    plt.scatter(
        projected_dataset_embeddings[:, 0],
        projected_dataset_embeddings[:, 1],
        s=10,
        color="gray",
        alpha=0.5,
        label="All Documents"
    )

    # Plot retrieved documents (green circles)
    plt.scatter(
        projected_retrieved_embeddings[:, 0],
        projected_retrieved_embeddings[:, 1],
        s=100,
        facecolors="none",
        edgecolors="g",
        linewidth=2,
        label="Retrieved Documents"
    )

    # Plot original query (red X)
    plt.scatter(
        projected_original_query_embedding[:, 0],
        projected_original_query_embedding[:, 1],
        s=150,
        marker="X",
        color="r",
        label="Original Query"
    )

    # Plot expanded query (orange X)
    plt.scatter(
        projected_augmented_query_embedding[:, 0],
        projected_augmented_query_embedding[:, 1],
        s=150,
        marker="X",
        color="orange",
        label="Augmented Query"
    )

    # Set chart properties
    plt.gca().set_aspect("equal", "datalim")
    plt.title(f"Query Embedding Space: {original_query}")
    plt.legend(loc="upper right")
    plt.axis("off")
    
    logger.info("Visualization complete")
    return plt

def run_rag_pipeline(
    pdf_path="data/microsoft-annual-report.pdf",
    query="What was the total profit for the year, and how does it compare to the previous year?",
    chunk_size=1000,
    token_size=256,
    n_results=5,
    model="gpt-4o-mini",
    show_visualization=True,
    debug=DEBUG
):
    """
    Run the complete RAG pipeline with query expansion
    
    Args:
        pdf_path (str): Path to the PDF document
        query (str): Query to search for
        chunk_size (int): Size of chunks for character-based splitting
        token_size (int): Size of chunks for token-based splitting
        n_results (int): Number of results to return
        model (str): OpenAI model to use for query expansion
        show_visualization (bool): Whether to display visualization
        debug (bool): Whether to print debug information
        
    Returns:
        tuple: (original results, augmented results)
    """
    # Initialize OpenAI client
    client = load_environment()
    
    # Process PDF document
    pdf_texts = process_document(pdf_path, debug)
    
    # Split text into chunks
    text_chunks = split_text(pdf_texts, chunk_size, token_size, debug)
    
    # Create vector database
    collection, embedding_function = create_vector_db(text_chunks)
    
    # Perform original query
    original_query_results = perform_query(collection, query, n_results)
    
    if debug:
        print("\nOriginal query results:")
        for doc in original_query_results["documents"][0]:
            print(word_wrap(doc))
            print("")
    
    # Generate augmented query
    hypothetical_answer = augment_query_generated(client, query, model)
    augmented_query = f"{query} {hypothetical_answer}"
    
    if debug:
        print("\nAugmented query:")
        print(word_wrap(augmented_query))
    
    # Perform augmented query
    augmented_query_results = perform_query(
        collection, augmented_query, n_results, is_augmented=True
    )
    
    if debug:
        print("\nAugmented query results:")
        for doc in augmented_query_results["documents"][0]:
            print(word_wrap(doc))
            print("")
    
    # Visualization if requested
    if show_visualization:
        # Get all embeddings
        all_embeddings = collection.get(include=["embeddings"])["embeddings"]
        
        # Create UMAP transform
        umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(all_embeddings)
        
        # Get query embeddings
        original_query_embedding = embedding_function([query])
        augmented_query_embedding = embedding_function([augmented_query])
        
        # Create visualization
        plt_figure = visualize_embeddings(
            all_embeddings,
            umap_transform,
            original_query_embedding,
            augmented_query_embedding,
            augmented_query_results["embeddings"][0],
            query
        )
        
        # Display the plot
        plt_figure.show()
    
    return original_query_results, augmented_query_results

if __name__ == "__main__":
    # Example usage
    run_rag_pipeline(
        pdf_path="data/microsoft-annual-report.pdf",
        query="What was the total profit for the year, and how does it compare to the previous year?",
        debug=True  # Enable debug output
    )



