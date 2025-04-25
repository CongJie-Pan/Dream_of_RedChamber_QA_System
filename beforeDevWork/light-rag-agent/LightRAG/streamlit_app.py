"""
Streamlit Web Interface for LightRAG Question Answering

This module provides a web interface for the LightRAG question answering system using Streamlit.
It allows users to ask questions about documents stored in a LightRAG knowledge base
and displays the answers with additional debugging information.

Features:
- Interactive web interface for asking questions
- Real-time streaming of responses
- Knowledge base status and statistics
- Support for switching between different knowledge bases
- Detailed error handling and logging
- Bilingual support (Chinese and English)

Usage:
    streamlit run streamlit_app.py

Author: CongJie Pan
Date: April 2025
"""

from dotenv import load_dotenv
import streamlit as st
import asyncio
import os
import sys
import logging
import traceback
from datetime import datetime
import json

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
from rag_agent import agent, RAGDeps, initialize_rag

# Configure logging with both file and console handlers
# This provides comprehensive logging for debugging and auditing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"lightrag_streamlit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("lightrag_streamlit")

# Load environment variables for API keys and other settings
load_dotenv()

# Define working directory for consistency with other modules
# This should match the directory used in the insert_pydantic_docs.py script
DEFAULT_WORKING_DIR = "./basicSinogyAsk"

async def get_agent_deps(working_dir=DEFAULT_WORKING_DIR):
    """
    Create a LightRAG instance and the agent dependencies.
    
    This function initializes the RAG system and prepares it for use by the agent.
    It's called during application startup and when switching knowledge bases.
    
    Args:
        working_dir (str): Working directory for the LightRAG instance
    
    Returns:
        RAGDeps: The dependencies required by the agent to perform RAG operations
    
    Raises:
        Exception: If initialization fails for any reason
    """
    try:
        # Initialize LightRAG with the appropriate configuration
        logger.info(f"Initializing LightRAG with working directory: {working_dir}")
        lightrag = await initialize_rag(working_dir)
        
        # Return the dependencies object needed by the agent
        logger.info("LightRAG initialized successfully, creating agent dependencies")
        return RAGDeps(lightrag=lightrag)
    except Exception as e:
        # Provide detailed error information for debugging
        error_details = traceback.format_exc()
        logger.error(f"Error initializing RAG: {e}")
        logger.error(f"Error details: {error_details}")
        st.error(f"Error initializing RAG: {str(e)}")
        st.error("Make sure you've run the insert_pydantic_docs.py script to populate the database.")
        raise

def get_knowledge_sources(working_dir=DEFAULT_WORKING_DIR):
    """
    Get information about the knowledge sources in the working directory.
    
    This function reads the metadata file created during document ingestion
    to provide statistics and information about the knowledge base.
    
    Args:
        working_dir (str): Working directory for the LightRAG instance
    
    Returns:
        dict: Information about the knowledge sources, including:
            - Number of files processed
            - Success/failure rates
            - Import timestamps
            - File details
    """
    try:
        # Check for metadata file created during document ingestion
        metadata_file = os.path.join(working_dir, "import_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            logger.info(f"Successfully loaded metadata from {metadata_file}")
            return metadata
        else:
            # Return basic information if no metadata file exists
            logger.warning(f"No metadata file found at {metadata_file}")
            return {
                "working_directory": os.path.abspath(working_dir),
                "exists": os.path.exists(working_dir),
                "no_metadata": True
            }
    except Exception as e:
        # Handle errors when reading metadata
        error_details = traceback.format_exc()
        logger.error(f"Error getting knowledge sources: {e}")
        logger.error(f"Error details: {error_details}")
        return {
            "error": str(e),
            "working_directory": os.path.abspath(working_dir) if working_dir else "Not specified"
        }

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
        with st.expander("Tool Call (Debug)", expanded=False):
            st.markdown(f"**Tool:** {part.name}")
            st.markdown(f"**Args:** {part.arguments}")
    # Tool returns - show in expander for debugging (hidden by default)
    elif part.part_kind == 'tool-return':
        with st.expander("Tool Return (Debug)", expanded=False):
            st.markdown(f"**Result:** {part.content}")


async def run_agent_with_streaming(user_input, working_dir=DEFAULT_WORKING_DIR):
    """
    Run the agent with streaming response.
    
    This function executes the question-answering agent and streams
    the response in real-time to the Streamlit interface, providing
    a better user experience for longer responses.
    
    Args:
        user_input (str): The user's question
        working_dir (str): Working directory for the LightRAG instance
        
    Yields:
        str: Streaming text chunks from the agent's response
    
    Raises:
        FileNotFoundError: If the working directory doesn't exist
    """
    # Verify the working directory exists before proceeding
    if not os.path.exists(working_dir):
        error_msg = f"Working directory {working_dir} not found. Please run insert_pydantic_docs.py first."
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
        working_dir = st.text_input("知識庫目錄", DEFAULT_WORKING_DIR)  # Knowledge Base Directory
        
        # If directory is changed, we need to reset the session
        if "current_working_dir" in st.session_state and working_dir != st.session_state.current_working_dir:
            if st.button("重新初始化系統"):  # Reinitialize System
                # Clear session state and force reinitialization
                st.session_state.pop("agent_deps", None)
                st.session_state.pop("messages", None)
                st.session_state.current_working_dir = working_dir
                st.experimental_rerun()
        
        # Display knowledge base statistics and status
        st.subheader("知識庫統計")  # Knowledge Base Statistics
        knowledge_info = get_knowledge_sources(working_dir)
        
        # Display appropriate status based on knowledge base state
        if not os.path.exists(working_dir):
            st.error(f"目錄不存在: {working_dir}")  # Directory does not exist
            st.warning("請先運行 insert_pydantic_docs.py 匯入數據")  # Please run import script first
        elif "error" in knowledge_info:
            st.error(f"讀取資料錯誤: {knowledge_info['error']}")  # Error reading data
        elif "no_metadata" in knowledge_info:
            st.warning("無元數據: 可能是舊版資料庫或未正確導入數據")  # No metadata: Possibly old database or import issues
            st.info(f"目錄存在: {working_dir}")  # Directory exists
        else:
            # Display success metrics when metadata is available
            st.success(f"成功導入文件: {knowledge_info.get('successful_files', 0)}/{knowledge_info.get('total_files', 0)}")  # Successfully imported files
            st.info(f"導入時間: {knowledge_info.get('start_time', 'Unknown')}")  # Import time
            
            # Show more details in an expandable section
            with st.expander("詳細資訊"):  # Detailed Information
                st.write(f"數據目錄: {knowledge_info.get('data_directory', 'Unknown')}")  # Data directory
                st.write(f"工作目錄: {knowledge_info.get('working_directory', 'Unknown')}")  # Working directory
                
                # List all imported files with their status
                if "files" in knowledge_info and knowledge_info["files"]:
                    st.subheader("已導入文件")  # Imported Files
                    for file in knowledge_info["files"]:
                        success = file.get("success", False)
                        icon = "✅" if success else "❌"
                        st.write(f"{icon} {file.get('file_name', 'Unknown')}")
                        if not success and "error" in file:
                            st.error(f"錯誤: {file['error']}")  # Error
    
    # Main content area - Chat interface
    st.title("📝 智能文本問答系統")  # Smart Text Q&A System
    
    # Information about the system
    with st.expander("關於本系統", expanded=False):  # About this system
        st.markdown("""
        這是一個基於 lightRAG (Retrieval-Augmented Generation) 的智能問答系統，可以針對已導入的文本資料回答問題。
        
        系統特點:
        - 使用 LightRAG 進行知識檢索
        - 使用 OpenAI GPT 模型生成回答
        - 支持中文和英文問答
        - 可以處理多種文本源
        - 主要詢問相關大一基本國學知識
        
        使用方法:
        1. 使用 insert_pydantic_docs.py 將文本資料導入知識庫
        2. 在側邊欄選擇知識庫目錄
        3. 在下方輸入框中提問
        """)
        # Translation:
        # This is a RAG (Retrieval-Augmented Generation) based intelligent Q&A system
        # that can answer questions about imported text data.
        #
        # System features:
        # - Uses LightRAG for knowledge retrieval
        # - Uses OpenAI GPT models for response generation
        # - Supports Chinese and English Q&A
        # - Can handle various text sources
        #
        # Usage:
        # 1. Use insert_pydantic_docs.py to import text data into the knowledge base
        # 2. Select knowledge base directory in the sidebar
        # 3. Ask questions in the input box below

    # Store the current working directory in session state for change detection
    if "current_working_dir" not in st.session_state:
        st.session_state.current_working_dir = working_dir

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize agent dependencies if not present
    if "agent_deps" not in st.session_state:
        with st.spinner("初始化 RAG 系統..."):  # Initializing RAG system...
            try:
                st.session_state.agent_deps = await get_agent_deps(working_dir)
                st.success("RAG 系統初始化成功!")  # RAG system initialized successfully!
            except Exception as e:
                st.error(f"初始化 RAG 系統失敗: {str(e)}")  # Failed to initialize RAG system
                st.error("請確保您已經執行 insert_pydantic_docs.py 導入數據。")  # Please make sure you've run the import script
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
                generator = run_agent_with_streaming(user_input, working_dir)
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
                if not os.path.exists(working_dir):
                    st.error(f"工作目錄不存在: {working_dir}")  # Working directory doesn't exist
                    st.error("請先執行 insert_pydantic_docs.py 導入數據")  # Please run the import script first


# Application entry point
if __name__ == "__main__":
    asyncio.run(main())
