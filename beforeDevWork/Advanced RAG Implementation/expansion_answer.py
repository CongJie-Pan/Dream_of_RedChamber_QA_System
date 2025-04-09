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
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
    OpenAIEmbeddingFunction
)
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

def split_text(pdf_texts, chunk_size=1000, use_chinese_segmentation=True, debug=DEBUG):
    """
    Split text using improved methods for better chunking, especially for Chinese text
    
    Args:
        pdf_texts (list): List of text content from PDF
        chunk_size (int): Size of chunks for character-based splitting
        use_chinese_segmentation (bool): Whether to use jieba for Chinese word segmentation
        debug (bool): Whether to print debug information
    
    Returns:
        list: List of text chunks after splitting
    """
    logger.info("Starting text splitting process")
    
    # Check if the text contains Chinese characters
    has_chinese = any(any('\u4e00' <= char <= '\u9fff' for char in text) for text in pdf_texts)
    
    # Use different separators and chunk settings based on text language
    if has_chinese:
        logger.info("Chinese text detected, using optimized splitting for Chinese")
        # Chinese text needs different separators that respect Chinese punctuation
        separators = ["。", "！", "？", "\n\n", "\n", ". ", " ", ""]
        # Use smaller chunks for Chinese text for better coherence
        actual_chunk_size = min(chunk_size, 500)
        # Add overlap to maintain context across chunks
        chunk_overlap = 50
        
        # Use jieba for Chinese word segmentation if requested
        if use_chinese_segmentation and has_chinese:
            try:
                import jieba
                logger.info("Using jieba for Chinese word segmentation")
                
                # Process each text chunk with jieba
                segmented_texts = []
                for text in pdf_texts:
                    if any('\u4e00' <= char <= '\u9fff' for char in text):
                        # Only apply jieba to text containing Chinese characters
                        words = jieba.cut(text)
                        segmented_text = " ".join(words)
                        segmented_texts.append(segmented_text)
                    else:
                        segmented_texts.append(text)
                
                # Replace the original texts with segmented ones
                pdf_texts = segmented_texts
                
                if debug:
                    print("\nSegmented Chinese text sample:")
                    print(word_wrap(pdf_texts[0][:200]))
            except ImportError:
                logger.warning("Jieba library not found. Chinese word segmentation disabled.")
            except Exception as e:
                logger.warning(f"Error during Chinese word segmentation: {str(e)}")
    else:
        logger.info("Non-Chinese text detected, using standard separators")
        separators = ["\n\n", "\n", ". ", " ", ""]
        actual_chunk_size = chunk_size
        chunk_overlap = 20
    
    # Use recursive character splitter with appropriate separators
    character_splitter = RecursiveCharacterTextSplitter(
        separators=separators, 
        chunk_size=actual_chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    # Split the text using the character splitter
    text_chunks = character_splitter.split_text("\n\n".join(pdf_texts))
    
    if debug:
        print(f"\nAfter character splitting - Total chunks: {len(text_chunks)}")
        print("Sample chunk (character split):")
        print(word_wrap(text_chunks[min(10, len(text_chunks)-1)]))
    
    # Skip SentenceTransformersTokenTextSplitter entirely as requested
    # This avoids [UNK] token issues with Chinese text when using transformer-based tokenizers
    logger.info(f"Text splitting complete. Generated {len(text_chunks)} chunks")
    logger.info(f"SentenceTransformersTokenTextSplitter step skipped to avoid [UNK] token issues")
    
    return text_chunks

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
    
    # Check if any of the chunks contain Chinese text
    has_chinese = any(any('\u4e00' <= char <= '\u9fff' for char in text) for text in text_chunks)
    
    # Create embedding function instance if not provided
    if embedding_function is None:
        # Check if OpenAI API key is available
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            # Always use OpenAI embeddings for Chinese text to avoid [UNK] tokens
            if has_chinese:
                logger.info("Chinese text detected, using OpenAI embedding function for better Chinese language support")
            else:
                logger.info("Using OpenAI embedding function as default")
                
            embedding_function = OpenAIEmbeddingFunction(
                api_key=openai_api_key,
                model_name="text-embedding-ada-002"
            )
        else:
            # Only use SentenceTransformer if OpenAI API key is not available and no Chinese text
            if has_chinese:
                logger.warning("Chinese text detected but OpenAI API key not found. Chinese may not embed properly.")
                logger.warning("Please set OPENAI_API_KEY for better Chinese language support.")
            
            logger.info("No OpenAI API key available, falling back to SentenceTransformer embedding function")
            embedding_function = SentenceTransformerEmbeddingFunction()
    elif has_chinese:
        # If embedding function was provided but text contains Chinese, check if it's OpenAI
        if not isinstance(embedding_function, OpenAIEmbeddingFunction):
            logger.warning("Chinese text detected but non-OpenAI embedding function provided.")
            logger.warning("This may cause [UNK] token issues with Chinese text.")
            logger.warning("Consider using OpenAI embeddings for better Chinese language support.")
    
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
    
    # Process the query with the same method used during retrieval if it's Chinese
    # Use the comprehensive processing to ensure consistency
    if has_chinese:
        logger.info("Chinese query detected, applying comprehensive Chinese text preprocessing for LLM")
        # Use the same comprehensive processing function used for queries and documents
        processed_query = process_chinese_text(query, use_segmentation=True)
        logger.info(f"Original query for LLM: '{query[:50]}...'")
        logger.info(f"Processed query for LLM: '{processed_query[:50]}...'")
    else:
        processed_query = query
    
    # Set system prompt for AI assistant based on detected language
    if has_chinese:
        prompt = """你是一位專業研究助理，能夠幫助用户分析文檔。         
        請針對用户的問題提供一個可能在文檔中出現的示例答案。         
        你的回答應該客觀、準確，並且遵循中文表達習慣。使用自然流暢的中文回答，不要使用機器翻譯的風格。         
        回答時無需聲明這是一個假設回答。如果回答中包含數字、日期或專有名詞，請確保它們是準確的格式。"""
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
            "content": processed_query if has_chinese else query
        },
    ]

    # Call OpenAI API to generate response
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    
    # Extract generated content from response
    content = response.choices[0].message.content
    
    # If we're dealing with Chinese, apply some post-processing to the generated answer
    if has_chinese:
        # Remove any [SEP] tokens that might have been included
        content = content.replace("[SEP]", "")
        
        # Normalize whitespace
        content = " ".join(content.split())
        
    logger.info("Query augmentation completed")
    return content

def process_chinese_text(text, use_segmentation=True):
    """
    Process Chinese text by applying jieba word segmentation and optimized splitting
    
    Args:
        text (str): The text to process
        use_segmentation (bool): Whether to use jieba for segmentation
        
    Returns:
        str: Processed text
    """
    # Check if text contains Chinese characters
    has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
    
    if not has_chinese:
        return text
    
    logger.info("Processing Chinese text with optimized methods")
    
    # Apply jieba word segmentation if requested
    if use_segmentation:
        try:
            # Use jieba for Chinese word segmentation
            import jieba
            logger.info("Using jieba to segment Chinese text")
            # Force loading of user dictionaries if any exist
            user_dict_path = os.path.join(os.path.dirname(__file__), "jieba_userdict.txt")
            if os.path.exists(user_dict_path):
                jieba.load_userdict(user_dict_path)
                logger.info(f"Loaded jieba user dictionary from {user_dict_path}")
            
            # Cut the text with jieba
            words = jieba.cut(text)
            text = " ".join(words)
            logger.info("Text segmented with jieba")
        except ImportError:
            logger.warning("Jieba library not found. Chinese word segmentation skipped.")
        except Exception as e:
            logger.warning(f"Error during Chinese word segmentation: {str(e)}")
    
    # Apply additional Chinese-specific processing
    
    # 1. Replace traditional Chinese punctuation with spaces for better tokenization
    chinese_punctuation = "，。！？；：""''「」【】《》（）～"
    for char in chinese_punctuation:
        text = text.replace(char, f" {char} ")
    
    # 2. Normalize whitespace
    text = " ".join(text.split())
    
    # 3. Add sentence boundary markers to help with context
    for char in "。！？":
        text = text.replace(f" {char} ", f" {char} [SEP] ")
    
    logger.info(f"Chinese text processed: '{text[:50]}...'")
    return text

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
    
    # Check if query contains Chinese characters
    has_chinese = any('\u4e00' <= char <= '\u9fff' for char in query_text)
    
    # Apply the same preprocessing to query as we do to documents
    if has_chinese:
        logger.info("Chinese query detected, applying comprehensive Chinese text preprocessing")
        
        # Always use text processing for Chinese to ensure consistency with document processing
        processed_query = process_chinese_text(query_text, use_segmentation=True)
        
        # Log the transformation for debugging
        if processed_query != query_text:
            logger.info(f"Original query: '{query_text[:50]}...'")
            logger.info(f"Processed query: '{processed_query[:50]}...'")
            logger.info("Query processing applied to match document processing")
        
        # Check if OpenAI embeddings are being used (strongly recommended for Chinese)
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
        if not isinstance(collection._embedding_function, OpenAIEmbeddingFunction):
            logger.warning("Chinese query detected but non-OpenAI embedding function is being used.")
            logger.warning("This may cause [UNK] token issues with Chinese text.")
            logger.warning("Consider using OpenAI embeddings for better Chinese language support.")
    else:
        processed_query = query_text
    
    # Perform the query with processed text
    try:
        results = collection.query(
            query_texts=[processed_query], 
            n_results=n_results,
            include=["documents", "embeddings"]
        )
        
        logger.info(f"Retrieved {len(results['documents'][0])} documents")
        return results
    except Exception as e:
        logger.error(f"Error during query: {str(e)}")
        
        # If error occurs with processed query, try again with original as fallback
        if processed_query != query_text:
            logger.warning("Error occurred with processed query. Trying again with original query as fallback.")
            results = collection.query(
                query_texts=[query_text], 
                n_results=n_results,
                include=["documents", "embeddings"]
            )
            logger.info(f"Retrieved {len(results['documents'][0])} documents using original query")
            return results
        else:
            # Re-raise the exception if we can't recover
            raise

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
    use_chinese_segmentation=True,
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
        use_chinese_segmentation (bool): Whether to use jieba for Chinese word segmentation
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
    text_chunks = split_text(pdf_texts, chunk_size, use_chinese_segmentation, debug)
    
    # Create vector database
    collection, embedding_function = create_vector_db(text_chunks)
    
    # Check for Chinese query and apply comprehensive processing
    has_chinese = any('\u4e00' <= char <= '\u9fff' for char in query)
    if has_chinese:
        logger.info("Chinese query detected in pipeline, applying comprehensive processing")
        processed_query = process_chinese_text(query, use_chinese_segmentation)
        if debug and processed_query != query:
            print(f"Original query: {query}")
            print(f"Processed query: {processed_query}")
    else:
        processed_query = query
    
    # Perform original query - use the processed query to ensure consistency with document processing
    original_query_results = perform_query(collection, processed_query, n_results)
    
    if debug:
        print("\nOriginal query results:")
        for doc in original_query_results["documents"][0]:
            print(word_wrap(doc))
            print("")
    
    # Generate augmented query
    hypothetical_answer = augment_query_generated(client, query, model)
    
    # Combine processed query with hypothetical answer to ensure consistency
    augmented_query = f"{processed_query} {hypothetical_answer}"
    
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
        
        # Get query embeddings - always use processed query for consistent embedding
        original_query_embedding = embedding_function([processed_query])
        augmented_query_embedding = embedding_function([augmented_query])
        
        # Create visualization
        plt_figure = visualize_embeddings(
            all_embeddings,
            umap_transform,
            original_query_embedding,
            augmented_query_embedding,
            augmented_query_results["embeddings"][0],
            query  # Keep original query for display purpose
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



