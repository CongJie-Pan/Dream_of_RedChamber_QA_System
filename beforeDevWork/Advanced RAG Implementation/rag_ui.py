"""
# ================================================================================================
# Advanced RAG System Interactive UI
# ================================================================================================
#
# This file implements an interactive UI for the Advanced RAG system using Streamlit.
# Features include:
#
# 1. PDF document upload and processing
# 2. Text splitting visualization and configuration
# 3. Interactive query with expansion options
# 4. Vector search results visualization
# 5. Embedding space visualization with UMAP
# 6. Multilingual support for documents and queries
#
# ================================================================================================
"""

import os
import io
import tempfile
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from expansion_answer import (
    load_environment,
    process_document,
    split_text,
    create_vector_db,
    augment_query_generated,
    perform_query,
    visualize_embeddings,
    process_chinese_text
)
from helper_utils import word_wrap
import umap
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction, OpenAIEmbeddingFunction

# Supported file types
SUPPORTED_FILE_TYPES = {
    'PDF': ['pdf'],
    'Word': ['docx'],
    'Text': ['txt'],
    'HTML': ['html', 'htm']
}
# Flatten supported file types for uploader
SUPPORTED_EXTENSIONS = [ext for exts in SUPPORTED_FILE_TYPES.values() for ext in exts]

# Available embedding models for different languages
EMBEDDING_MODELS = {
    "OpenAI": "text-embedding-ada-002",  # Best for all languages, especially Chinese
    "English": "sentence-transformers/all-MiniLM-L6-v2",
    "Multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "Chinese": "sentence-transformers/distiluse-base-multilingual-cased-v2"
}

# Page configuration
st.set_page_config(
    page_title="Advanced RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for configuration
with st.sidebar:
    st.title("🔍 Advanced RAG System")
    st.markdown("---")
    
    # Model selection
    st.subheader("Model Configuration")
    openai_model = st.selectbox(
        "OpenAI Model",
        ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o"],
        index=0,
        help="Select the OpenAI model to use for query expansion"
    )
    
    # Embedding model selection
    st.subheader("Embedding Model")
    embedding_model_name = st.selectbox(
        "Language Model",
        list(EMBEDDING_MODELS.keys()),
        index=0,  # Default to OpenAI model
        help="Select language model for embeddings - OpenAI recommended for all languages, especially Chinese"
    )
    embedding_model_path = EMBEDDING_MODELS[embedding_model_name]
    
    # Text splitting parameters
    st.subheader("Text Splitting")
    chunk_size = st.slider(
        "Character Chunk Size",
        min_value=300,
        max_value=2000,
        value=800,
        step=100,
        help="Size of chunks for character-based splitting (smaller values recommended for Chinese)"
    )
    
    use_chinese_segmentation = st.checkbox(
        "Enable Chinese Word Segmentation", 
        value=True,
        help="Uses jieba to improve Chinese text segmentation (recommended for Chinese documents)"
    )
    
    # Retrieval parameters
    st.subheader("Retrieval Settings")
    n_results = st.slider(
        "Number of Results",
        min_value=1,
        max_value=20, 
        value=5,
        help="Number of documents to retrieve"
    )
    
    # Debug mode
    debug_mode = st.checkbox("Debug Mode", value=False, help="Enable detailed logging")
    
    # Language settings info
    st.subheader("Language Support")
    st.markdown("""
    - **OpenAI**: Recommended for all languages, especially Chinese (avoids [UNK] token issues)
    - **English**: Works best with English documents only
    - **Multilingual**: Supports multiple languages but may have issues with some Chinese characters
    - **Chinese**: Basic Chinese support, may have [UNK] token issues with some characters
    """)
    
    # Add warning about model dimension compatibility
    st.warning("""
    **Important**: If you change the embedding model after processing a document, 
    you need to click "Reprocess with Current Model" button to avoid dimension mismatch errors.
    Different models use different vector dimensions:
    - OpenAI: 1536 dimensions
    - English: 384 dimensions
    - Multilingual: 384 dimensions
    - Chinese: 512 dimensions
    
    The system now uses direct character splitting without token-based splitting for all text.
    This significantly improves handling of Chinese text by avoiding tokenization issues.
    OpenAI's embedding model provides the best results for Chinese text.
    """)
    
    # Supported file types information
    st.subheader("Supported File Types")
    for file_type, extensions in SUPPORTED_FILE_TYPES.items():
        st.markdown(f"- **{file_type}**: {', '.join([f'.{ext}' for ext in extensions])}")
    
    st.markdown("---")
    st.markdown("Made with ❤️ by Advanced RAG Team")

# Main app content
st.title("Advanced RAG System with Query Expansion")
st.markdown("""
This system demonstrates advanced RAG techniques including:
- Multi-level text chunking
- Query expansion using AI
- Vector embedding visualization
- Support for multiple document formats
- Multilingual document processing
""")

# Initialize session state
if 'openai_client' not in st.session_state:
    try:
        st.session_state.openai_client = load_environment()
        st.success("OpenAI API connection established!")
    except Exception as e:
        st.error(f"Error connecting to OpenAI API: {str(e)}")
        st.info("Please ensure your OpenAI API key is set in the .env file")
        st.stop()

if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False
    
if 'text_chunks' not in st.session_state:
    st.session_state.text_chunks = None
    
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
    
if 'embedding_function' not in st.session_state:
    st.session_state.embedding_function = None

# Document Upload Section
st.header("Step 1: Upload Document")
uploaded_file = st.file_uploader("Upload a document", type=SUPPORTED_EXTENSIONS)

if uploaded_file is not None:
    # Display file information
    file_type = uploaded_file.type
    file_size = uploaded_file.size / 1024  # KB
    
    # Show file details in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**File Name**: {uploaded_file.name}")
    with col2:
        st.info(f"**File Size**: {file_size:.2f} KB")
    with col3:
        st.info(f"**Selected Language Model**: {embedding_model_name}")
    
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
        tmp.write(uploaded_file.getvalue())
        temp_path = tmp.name
    
    # Process document when user clicks the button
    if st.button("Process Document"):
        with st.spinner(f"Processing {uploaded_file.name} with {embedding_model_name} model..."):
            try:
                # Process document text
                text_sections = process_document(temp_path, debug=debug_mode)
                st.session_state.section_count = len(text_sections)
                
                # Text splitting
                text_chunks = split_text(text_sections, chunk_size, use_chinese_segmentation, debug=debug_mode)
                st.session_state.text_chunks = text_chunks
                
                # Create a unique collection name based on file name and embedding model
                # This ensures different models don't try to use the same collection with different dimensions
                model_suffix = embedding_model_name.split()[0].lower()
                # Remove any invalid characters from collection name
                safe_model_suffix = ''.join(c if c.isalnum() or c == '_' or c == '-' else '_' for c in model_suffix)
                safe_filename = ''.join(c if c.isalnum() or c == '_' or c == '-' else '_' for c in uploaded_file.name.replace('.', '_').replace(' ', '_'))
                collection_name = f"collection_{safe_filename}_{safe_model_suffix}"
                st.session_state.collection_name = collection_name
                
                try:
                    # Initialize embedding function with the selected model
                    if embedding_model_name == "OpenAI":
                        embedding_function = OpenAIEmbeddingFunction(
                            api_key=os.getenv("OPENAI_API_KEY"),
                            model_name=embedding_model_path
                        )
                    else:
                        embedding_function = SentenceTransformerEmbeddingFunction(model_name=embedding_model_path)
                    
                    st.session_state.embedding_function = embedding_function
                    
                    vector_db, _ = create_vector_db(text_chunks, collection_name=collection_name, 
                                                   embedding_function=embedding_function)
                    st.session_state.vector_db = vector_db
                    
                    st.session_state.document_processed = True
                    st.success(f"Document processed successfully! Extracted {len(text_sections)} sections and created {len(text_chunks)} text chunks.")
                    
                    # Add a note about persistence
                    st.info("Your document has been indexed and will be available for future queries even if you restart the application.")
                except Exception as e:
                    st.error(f"Error creating vector database: {str(e)}")
                
                # Display sample chunks to verify encoding
                with st.expander("View Sample Text Chunks"):
                    sample_size = min(5, len(text_chunks))
                    for i in range(sample_size):
                        st.text_area(f"Chunk {i+1}", text_chunks[i], height=100)
            
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
                st.info("Please make sure your document is not corrupted and is in one of the supported formats.")
else:
    # Show supported formats
    st.markdown("""
    ### Please upload one of these supported document types:
    - PDF documents (.pdf)
    - Word documents (.docx)
    - Plain text files (.txt)
    - HTML files (.html, .htm)
    """)
    st.info("👆 Upload a document to get started")
    
    # Language selection note
    st.warning("""
    **Note for Chinese documents**: The system now uses improved Chinese text handling:
    
    1. OpenAI embedding model is selected by default (recommended for Chinese)
    2. Character-based splitting is used instead of token-based splitting
    3. Chinese punctuation and separators are properly recognized
    4. Jieba word segmentation is enabled by default for better Chinese text chunking
    
    These changes will help avoid [UNK] token issues with Chinese text.
    """)

# Query Section (only show if document is processed)
if st.session_state.document_processed:
    st.header("Step 2: Ask Questions")
    
    query = st.text_input("Enter your query", placeholder="e.g., What was the total revenue for the year?")
    
    use_query_expansion = st.checkbox("Use Query Expansion", value=True, 
                                    help="Use AI to expand your query for better results")
    
    show_visualization = st.checkbox("Show Embedding Visualization", value=True,
                                    help="Visualize document and query embeddings")
    
    if st.button("Search"):
        if not query:
            st.warning("Please enter a query")
        else:
            try:
                with st.spinner("Searching..."):
                    # Original query
                    st.subheader("Original Query")
                    st.info(query)
                    
                    # Perform original query with error handling
                    try:
                        # Check if query contains Chinese and preprocess if needed
                        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in query)
                        if has_chinese and use_chinese_segmentation:
                            st.info("中文查詢已檢測到！正在應用優化的中文處理...")
                            st.info("Chinese query detected, applying optimized Chinese processing...")
                            
                            # Display original query
                            with st.expander("View original vs processed query"):
                                # Process the query with the same method used for document indexing
                                processed_query = process_chinese_text(query, use_segmentation=True)
                                st.text("Original query:")
                                st.code(query)
                                st.text("Processed query (used for searching):")
                                st.code(processed_query)
                                st.info("The processed version is used for searching to match how documents were indexed.")
                        
                        # Perform query (preprocessing is handled within perform_query)
                        original_results = perform_query(
                            st.session_state.vector_db, 
                            query, 
                            n_results=n_results
                        )
                        
                        # Display original results
                        st.markdown("#### Original Results")
                        for i, doc in enumerate(original_results["documents"][0]):
                            with st.expander(f"Result {i+1}"):
                                st.markdown(doc)
                    except Exception as e:
                        error_message = str(e)
                        st.error(f"Error performing original query: {error_message}")
                        
                        # Special handling for dimension mismatch errors
                        if "dimension" in error_message.lower() and "match" in error_message.lower():
                            st.warning("""
                            ### Dimension Mismatch Detected
                            
                            This error occurs when you've switched to a different embedding model after processing a document.
                            Different models create vectors with different dimensions, which causes this conflict.
                            
                            **Solution:**
                            1. Scroll down to the bottom of the page
                            2. Click the "Reprocess with Current Model" button in the Database Management section
                            3. Process the document again with your current model selection
                            4. Try your query again
                            """)
                        
                        st.stop()
                    
                    # Query expansion if enabled
                    if use_query_expansion:
                        with st.spinner("Expanding query..."):
                            st.subheader("Expanded Query")
                            
                            try:
                                # Generate hypothetical answer
                                hypothetical_answer = augment_query_generated(
                                    st.session_state.openai_client, 
                                    query, 
                                    model=openai_model
                                )
                                
                                # Get processed query if Chinese text detected
                                has_chinese = any('\u4e00' <= char <= '\u9fff' for char in query)
                                if has_chinese and use_chinese_segmentation:
                                    processed_query = process_chinese_text(query, use_segmentation=True)
                                    # Combine processed query with hypothetical answer for better matching
                                    expanded_query = f"{processed_query} {hypothetical_answer}"
                                else:
                                    expanded_query = f"{query} {hypothetical_answer}"
                                
                                # Show expanded query
                                with st.expander("View AI-generated expansion"):
                                    st.info("Hypothetical answer generated by AI:")
                                    st.markdown(hypothetical_answer)
                                
                                # Check if expanded query contains Chinese
                                if any('\u4e00' <= char <= '\u9fff' for char in expanded_query) and use_chinese_segmentation:
                                    st.info("中文內容在擴展查詢中被檢測到，正在應用優化處理...")
                                    st.info("Chinese text detected in expanded query, applying optimized processing...")
                                    
                                    with st.expander("View processed expanded query"):
                                        # Process expanded query for consistency
                                        processed_expanded_query = process_chinese_text(expanded_query, use_segmentation=True)
                                        st.text("Original expanded query:")
                                        st.code(expanded_query)
                                        st.text("Processed expanded query (used for searching):")
                                        st.code(processed_expanded_query)
                                    
                                    # Use processed expanded query for display
                                    display_expanded_query = processed_expanded_query
                                else:
                                    display_expanded_query = expanded_query
                                    
                                st.info("Expanded query:")
                                st.markdown(display_expanded_query)
                                
                                # Perform search with expanded query
                                if any('\u4e00' <= char <= '\u9fff' for char in expanded_query) and use_chinese_segmentation:
                                    # Ensure the expanded query used for search is correctly processed
                                    processed_expanded_query = process_chinese_text(expanded_query, use_segmentation=True)
                                    expanded_results = perform_query(
                                        st.session_state.vector_db, 
                                        processed_expanded_query, 
                                        n_results=n_results,
                                        is_augmented=True
                                    )
                                else:
                                    expanded_results = perform_query(
                                        st.session_state.vector_db, 
                                        expanded_query, 
                                        n_results=n_results,
                                        is_augmented=True
                                    )
                                
                                # Display expanded results
                                st.markdown("#### Expanded Query Results")
                                for i, doc in enumerate(expanded_results["documents"][0]):
                                    with st.expander(f"Result {i+1}"):
                                        st.markdown(doc)
                                
                                # Compare similarities between results
                                original_docs = set(original_results["documents"][0])
                                expanded_docs = set(expanded_results["documents"][0])
                                common_docs = original_docs.intersection(expanded_docs)
                                
                                st.markdown("#### Result Comparison")
                                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                                with metrics_col1:
                                    st.metric("Original Results", len(original_docs))
                                with metrics_col2:
                                    st.metric("Expanded Results", len(expanded_docs))
                                with metrics_col3:
                                    st.metric("Common Results", len(common_docs))
                                
                                # Display visualization
                                if show_visualization:
                                    with st.spinner("Generating visualization..."):
                                        st.subheader("Embedding Space Visualization")
                                        
                                        # Get all embeddings
                                        all_embeddings = st.session_state.vector_db.get(include=["embeddings"])["embeddings"]
                                        
                                        # Create UMAP transform
                                        umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(all_embeddings)
                                        
                                        # Check if query contains Chinese characters
                                        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in query)
                                        if has_chinese and use_chinese_segmentation:
                                            # Process the query with the same method used during retrieval
                                            processed_query = process_chinese_text(query, use_segmentation=True)
                                            # Use processed query for embedding to match how documents were processed
                                            original_query_embedding = st.session_state.embedding_function([processed_query])
                                            
                                            # Also process the expanded query if it contains Chinese
                                            if any('\u4e00' <= char <= '\u9fff' for char in expanded_query):
                                                processed_expanded_query = process_chinese_text(expanded_query, use_segmentation=True)
                                                augmented_query_embedding = st.session_state.embedding_function([processed_expanded_query])
                                                
                                                # Add explanation about preprocessing for visualization
                                                with st.expander("Embedding processing explanation"):
                                                    st.info("Chinese text detected. For visualization, both the original query and expanded query have been processed to match document embeddings.")
                                                    st.text("Query embedding created from the processed version:")
                                                    st.code(processed_query)
                                                    st.text("Expanded query embedding created from the processed version:")
                                                    st.code(processed_expanded_query)
                                            else:
                                                augmented_query_embedding = st.session_state.embedding_function([expanded_query])
                                        else:
                                            # For non-Chinese queries, use as-is
                                            original_query_embedding = st.session_state.embedding_function([query])
                                            augmented_query_embedding = st.session_state.embedding_function([expanded_query])
                                        
                                        # Create visualization as a matplotlib figure
                                        fig = visualize_embeddings(
                                            all_embeddings,
                                            umap_transform,
                                            original_query_embedding,
                                            augmented_query_embedding,
                                            expanded_results["embeddings"][0],
                                            query
                                        )
                                        
                                        # Display the figure
                                        st.pyplot(fig)
                                        
                                        # Add a download button for the figure
                                        buf = io.BytesIO()
                                        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                                        buf.seek(0)
                                        st.download_button(
                                            label="Download Visualization",
                                            data=buf,
                                            file_name="embedding_visualization.png",
                                            mime="image/png"
                                        )
                            except Exception as e:
                                st.error(f"Error during query expansion: {str(e)}")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
                st.info("Try refreshing the page and processing the document again.")
else:
    st.info("👆 Upload and process a document first")
    
# Add instructions at the bottom
with st.expander("How to Use This Tool"):
    st.markdown("""
    ### Instructions
    
    1. **Upload Document**: Upload any supported document you want to analyze.
    2. **Process Document**: Click the "Process Document" button to extract text and create embeddings.
    3. **Ask Questions**: Type your question and select whether to use query expansion.
    4. **Analyze Results**: Compare original results with expanded query results.
    5. **Visualize**: Explore the embedding space visualization to understand document relationships.
    
    ### About Query Expansion
    
    Query expansion uses AI to generate a hypothetical answer to your question, then combines it with your original query.
    This helps retrieve more relevant documents by expanding the semantic meaning of your search.
    
    ### Language Support
    
    For Chinese documents:
    - System now uses character-based splitting optimized for Chinese text
    - OpenAI embedding model is recommended and set as default
    - Chinese punctuation is properly recognized as text separators
    - Jieba word segmentation improves Chinese text processing
    - Both Traditional and Simplified Chinese characters are supported
    
    ### Visualization Explained
    
    - **Gray Dots**: All document chunks in the corpus
    - **Green Circles**: Retrieved documents
    - **Red X**: Original query
    - **Orange X**: Expanded query
    
    Documents closest to the query points are most semantically similar.
    """)

# Add a section to manage collections
if st.session_state.document_processed:
    st.markdown("---")
    st.header("Database Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Reset button to clear state and allow uploading a new document
        if st.button("Reset and Upload New Document"):
            # Reset all session state variables
            for key in list(st.session_state.keys()):
                if key != 'openai_client':  # Keep the OpenAI client
                    del st.session_state[key]
                    
            st.session_state.document_processed = False
            st.success("Reset successful! You can now upload a new document.")
            st.experimental_rerun()
    
    with col2:
        # Button to reprocess current document with a different embedding model
        if st.button("Reprocess with Current Model"):
            st.session_state.document_processed = False
            # Keep the uploaded file but reset processing state
            st.success(f"Ready to reprocess document with {embedding_model_name} model.")
            st.experimental_rerun()
    
    with col3:
        # Information about persistence
        st.info(f"Current collection: **{st.session_state.get('collection_name', 'None')}** with **{st.session_state.vector_db.count() if st.session_state.get('vector_db') else 0}** documents")

# Check if Streamlit is run
if __name__ == "__main__":
    pass  # Streamlit already executes the code when run 