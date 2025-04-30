"""
Streamlit Web Interface for BasicRAG Question Answering

This module provides a web interface for the BasicRAG question answering system using Streamlit.
It allows users to ask questions about documents stored in a ChromaDB knowledge base
and displays the answers with additional debugging information.

Features:
- Interactive web interface for asking questions
- Real-time streaming of responses
- Knowledge base status and statistics
- Support for switching between different collections
- Detailed error handling and logging
- Bilingual support (Chinese and English)

Usage:
    streamlit run streamlit_app.py

Author: CongJie Pan (adapted for BasicRAG)
Date: April 2023
"""

import os
import sys
import asyncio
import logging
import traceback
from datetime import datetime
import json
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st

# Import all the message part classes from pydantic_ai for structured communication
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)

# Import from the rag_agent module for question answering
from rag_agent import agent, RAGDeps, get_import_metadata, DEFAULT_DB_DIR, DEFAULT_COLLECTION, DEFAULT_EMBEDDING_MODEL
from utils import get_chroma_client, get_or_create_collection

# Configure logging with timestamps and appropriate formats
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"streamlit_app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("streamlit_app")

# Load environment variables for API keys
load_dotenv()

# Check for OpenAI API key at module import time
if not os.getenv("OPENAI_API_KEY"):
    logger.error("OPENAI_API_KEY environment variable not set.")
    logger.error("Please create a .env file with your OpenAI API key or set it in your environment.")
    sys.exit(1)

async def get_agent_deps(db_dir=DEFAULT_DB_DIR, collection_name=DEFAULT_COLLECTION, embedding_model=DEFAULT_EMBEDDING_MODEL):
    """
    Create a ChromaDB client and agent dependencies.
    
    This function initializes the RAG system dependencies and prepares it for use by the agent.
    It's called during application startup and when switching collections.
    
    Args:
        db_dir (str): Database directory for ChromaDB
        collection_name (str): Name of the ChromaDB collection to use
        embedding_model (str): Name of the embedding model to use
    
    Returns:
        RAGDeps: The dependencies required by the agent to perform RAG operations
    
    Raises:
        Exception: If initialization fails for any reason
    """
    try:
        # Initialize ChromaDB client with the appropriate configuration
        logger.info(f"Initializing ChromaDB client from directory: {db_dir}")
        chroma_client = get_chroma_client(db_dir)
        
        # Get or create the specified collection
        collection = get_or_create_collection(
            chroma_client,
            collection_name,
            embedding_model_name=embedding_model
        )
        
        # Return the dependencies object needed by the agent
        logger.info("ChromaDB client initialized successfully, creating agent dependencies")
        return RAGDeps(
            chroma_client=chroma_client,
            collection_name=collection_name,
            embedding_model=embedding_model
        )
    except Exception as e:
        # Provide detailed error information for debugging
        error_details = traceback.format_exc()
        logger.error(f"Error initializing ChromaDB: {e}")
        logger.error(f"Error details: {error_details}")
        st.error(f"Error initializing ChromaDB: {str(e)}")
        st.error("確保您已運行 insert_docs.py 腳本來導入數據。")  # Make sure you've run the insert_docs.py script
        raise

def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    
    This function handles different types of message parts and displays them
    appropriately in the Streamlit interface, including user messages,
    assistant responses, tool calls, and tool returns.
    
    Args:
        part: A message part (user prompt, text, tool call, etc.)
    """
    # User prompt - display in user chat bubble
    if part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    # Text response from assistant - display in assistant chat bubble
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)            
    # Tool calls - show in expander for debugging (hidden by default)
    elif part.part_kind == 'tool-call':
        with st.expander("工具呼叫 (調試)", expanded=False):  # Tool Call (Debug)
            st.markdown(f"**工具:** {part.name}")  # Tool
            st.markdown(f"**參數:** {part.arguments}")  # Arguments
    # Tool returns - show in expander for debugging (hidden by default)
    elif part.part_kind == 'tool-return':
        with st.expander("工具返回 (調試)", expanded=False):  # Tool Return (Debug)
            st.markdown(f"**結果:** {part.content}")  # Result

async def run_agent_with_streaming(user_input, db_dir=DEFAULT_DB_DIR, collection_name=DEFAULT_COLLECTION, embedding_model=DEFAULT_EMBEDDING_MODEL):
    """
    Run the agent with streaming response.
    
    This function executes the question-answering agent and streams
    the response in real-time to the Streamlit interface, providing
    a better user experience for longer responses.
    
    Args:
        user_input (str): The user's question
        db_dir (str): Database directory for ChromaDB
        collection_name (str): Name of the ChromaDB collection to use
        embedding_model (str): Name of the embedding model to use
        
    Yields:
        str: Streaming text chunks from the agent's response
    
    Raises:
        FileNotFoundError: If the database directory doesn't exist
    """
    # Verify the database directory exists before proceeding
    if not os.path.exists(db_dir):
        error_msg = f"數據庫目錄不存在: {db_dir}。請先運行 insert_docs.py 來導入數據。"  # Database directory doesn't exist
        logger.error(error_msg)
        st.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    logger.info(f"Running agent with input: {user_input}")
    
    # Run the agent with streaming response
    async with agent.run_stream(
        user_input, deps=st.session_state.agent_deps, message_history=st.session_state.messages
    ) as result:
        # Stream the response chunks as they become available
        async for message in result.stream_text(delta=True):  
            logger.debug(f"Streaming chunk: {message[:20]}...")
            yield message

    # Add the new messages to the chat history for context preservation
    st.session_state.messages.extend(result.new_messages())
    logger.info("Agent run completed")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~ Main Function with UI Creation ~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

async def main():
    """
    Main function that creates and runs the Streamlit interface.
    
    This function:
    1. Sets up the page layout and configuration
    2. Creates the sidebar for system settings
    3. Sets up the main chat interface
    4. Handles user inputs and agent responses
    5. Manages session state for conversation history
    """
    # Set page title, icon, and layout options
    st.set_page_config(
        page_title="智能文本問答系統",  # Smart Text Q&A System
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar for configuration and system settings
    with st.sidebar:
        st.title("📚 系統設定")  # System Settings
        
        # Database directory selection
        db_dir = st.text_input("數據庫目錄", DEFAULT_DB_DIR)  # Database Directory
        
        # Collection selection
        collection_name = st.text_input("集合名稱", DEFAULT_COLLECTION)  # Collection Name
        
        # Embedding model selection
        embedding_model = st.text_input("嵌入模型", DEFAULT_EMBEDDING_MODEL)  # Embedding Model
        
        # If configuration is changed, we need to reset the session
        if ("current_db_dir" in st.session_state and db_dir != st.session_state.current_db_dir) or \
           ("current_collection" in st.session_state and collection_name != st.session_state.current_collection) or \
           ("current_embedding_model" in st.session_state and embedding_model != st.session_state.current_embedding_model):
            if st.button("重新初始化系統"):  # Reinitialize System
                # Clear session state and force reinitialization
                st.session_state.pop("agent_deps", None)
                st.session_state.pop("messages", None)
                st.session_state.current_db_dir = db_dir
                st.session_state.current_collection = collection_name
                st.session_state.current_embedding_model = embedding_model
                st.experimental_rerun()
        
        # Display knowledge base statistics and status
        st.subheader("數據庫統計")  # Database Statistics
        metadata = get_import_metadata(db_dir)
        
        # Display appropriate status based on knowledge base state
        if not os.path.exists(db_dir):
            st.error(f"目錄不存在: {db_dir}")  # Directory does not exist
            st.warning("請先運行 insert_docs.py 匯入數據")  # Please run import script first
        elif "error" in metadata:
            st.error(f"讀取資料錯誤: {metadata['error']}")  # Error reading data
        else:
            # Display success metrics when metadata is available
            if "total_chunks_added" in metadata:
                st.success(f"已成功導入區塊: {metadata.get('total_chunks_added', 0)}")  # Successfully imported chunks
            
            if "successful_files" in metadata:
                st.success(f"成功導入文件: {metadata.get('successful_files', 0)}/{metadata.get('total_files', 0)}")  # Successfully imported files
            
            if "successful_urls" in metadata:
                st.success(f"成功導入URL: {metadata.get('successful_urls', 0)}/{metadata.get('total_urls', 0)}")  # Successfully imported URLs
            
            if "start_time" in metadata:
                st.info(f"導入時間: {metadata.get('start_time', 'Unknown')}")  # Import time
            
            # Show more details in an expandable section
            with st.expander("詳細資訊"):  # Detailed Information
                st.write(f"數據目錄: {metadata.get('db_directory', 'Unknown')}")  # Data directory
                st.write(f"集合名稱: {metadata.get('collection_name', 'Unknown')}")  # Collection name
                st.write(f"嵌入模型: {metadata.get('embedding_model', 'Unknown')}")  # Embedding model
                
                # List all imported files with their status
                if "files" in metadata and metadata["files"]:
                    st.subheader("已導入文件")  # Imported Files
                    for file in metadata["files"]:
                        success = file.get("success", False)
                        icon = "✅" if success else "❌"
                        st.write(f"{icon} {file.get('file_name', 'Unknown')}")
                        if not success and "error" in file:
                            st.error(f"錯誤: {file['error']}")  # Error
                
                # List all imported URLs with their status
                if "urls" in metadata and metadata["urls"]:
                    st.subheader("已導入URL")  # Imported URLs
                    for url in metadata["urls"]:
                        success = url.get("success", False)
                        icon = "✅" if success else "❌"
                        st.write(f"{icon} {url.get('url', 'Unknown')}")
                        if not success and "error" in url:
                            st.error(f"錯誤: {url['error']}")  # Error
    
    # Main content area - Chat interface
    st.title("📝 智能文本問答系統")  # Smart Text Q&A System
    
    # Information about the system
    with st.expander("關於本系統", expanded=False):  # About this system
        st.markdown("""
        這是一個基於 RAG (Retrieval-Augmented Generation) 的智能問答系統，可以針對已導入的文本資料回答問題。
        
        系統特點:
        - Agentic RAG 系統
        - 使用 ChromaDB 進行知識檢索
        - 使用 OpenAI GPT 模型生成回答
        - 支持中文和英文問答
        - 可以處理多種文本源
        
        使用方法:
        1. 使用 insert_docs.py 將文本資料導入知識庫
        2. 在側邊欄選擇數據庫目錄和集合名稱
        3. 在下方輸入框中提問
        """)
        # Translation:
        # This is a RAG (Retrieval-Augmented Generation) based intelligent Q&A system
        # that can answer questions about imported text data.
        #
        # System features:
        # - Uses ChromaDB for knowledge retrieval
        # - Uses OpenAI GPT models for response generation
        # - Supports Chinese and English Q&A
        # - Can handle various text sources
        #
        # Usage:
        # 1. Use insert_docs.py to import text data into the knowledge base
        # 2. Select database directory and collection in the sidebar
        # 3. Ask questions in the input box below

    # Store the current configuration in session state for change detection
    if "current_db_dir" not in st.session_state:
        st.session_state.current_db_dir = db_dir
    if "current_collection" not in st.session_state:
        st.session_state.current_collection = collection_name
    if "current_embedding_model" not in st.session_state:
        st.session_state.current_embedding_model = embedding_model

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize agent dependencies if not present
    if "agent_deps" not in st.session_state:
        with st.spinner("初始化 RAG 系統..."):  # Initializing RAG system...
            try:
                st.session_state.agent_deps = await get_agent_deps(
                    db_dir, 
                    collection_name,
                    embedding_model
                )
                st.success("RAG 系統初始化成功!")  # RAG system initialized successfully!
            except Exception as e:
                st.error(f"初始化 RAG 系統失敗: {str(e)}")  # Failed to initialize RAG system
                st.error("請確保您已執行 insert_docs.py 導入數據。")  # Please make sure you've run the import script
                return

    # Display all messages from the conversation so far
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    # Chat input for the user
    user_input = st.chat_input("輸入您的問題...")  # Enter your question...

    # Process user input when submitted
    if user_input:
        logger.info(f"User input: {user_input}")
        
        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display the assistant's response with streaming
        with st.chat_message("assistant"):
            # Create a placeholder for the streaming text
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Consume the async generator to get streaming text chunks
                generator = run_agent_with_streaming(
                    user_input, 
                    db_dir, 
                    collection_name,
                    embedding_model
                )
                async for message in generator:
                    full_response += message
                    # Display with a blinking cursor (▌) to indicate typing
                    message_placeholder.markdown(full_response + "▌")
                
                # Final response without the cursor
                message_placeholder.markdown(full_response)
                logger.info("Response completed successfully")
            except Exception as e:
                # Handle and display errors
                error_details = traceback.format_exc()
                error_msg = f"生成回應時出錯: {str(e)}"  # Error generating response
                logger.error(error_msg)
                logger.error(f"Error details: {error_details}")
                st.error(error_msg)
                
                # If directory doesn't exist, provide specific guidance
                if not os.path.exists(db_dir):
                    st.error(f"數據庫目錄不存在: {db_dir}")  # Database directory doesn't exist
                    st.error("請先執行 insert_docs.py 導入數據")  # Please run the import script first


# Application entry point
if __name__ == "__main__":
    asyncio.run(main())
