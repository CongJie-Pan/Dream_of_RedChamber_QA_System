import os
import time
import glob
import streamlit as st
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import datetime
import json

# Setup logging
logger.remove()
logger.add(
    "debug.log",
    format="{time} {level} {message}",
    level="DEBUG",
    rotation="10 MB",
    retention="1 week"
)
logger.add(lambda msg: st.sidebar.write(f"LOG: {msg}") if "streamlit_debug" in st.session_state and st.session_state.streamlit_debug else None)

# Load environment variables
load_dotenv()

# Constants
DATA_DIR = r"D:\AboutCoding\PKUDH_Project\DreamOf_RedMansions\QA_System\beforeDevWork\light-rag-agent\LightRAG\data"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# Initialize session state variables
if "converted_files" not in st.session_state:
    st.session_state.converted_files = []
if "current_file" not in st.session_state:
    st.session_state.current_file = ""
if "remaining_files" not in st.session_state:
    st.session_state.remaining_files = []
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "converted_text" not in st.session_state:
    st.session_state.converted_text = ""
if "streamlit_debug" not in st.session_state:
    st.session_state.streamlit_debug = False
if "api_connected" not in st.session_state:
    st.session_state.api_connected = False

def get_all_text_files():
    """
    Get all text files from the data directory.
    
    Returns:
        list: List of text file paths
    """
    logger.debug(f"Searching for text files in {DATA_DIR}")
    files = glob.glob(os.path.join(DATA_DIR, "*.txt"))
    logger.info(f"Found {len(files)} text files")
    return files

def test_api_connection():
    """
    Test the connection to the DeepSeek API.
    
    Returns:
        bool: True if connection is successful, False otherwise
    """
    try:
        logger.debug("Testing DeepSeek API connection")
        if not DEEPSEEK_API_KEY:
            logger.error("DeepSeek API key is not set")
            return False
        
        # modify the initialization method to avoid the proxies parameter problem
        try:
            client = OpenAI(
                api_key=DEEPSEEK_API_KEY, 
                base_url="https://api.deepseek.com"
            )
        except TypeError as e:
            if "unexpected keyword argument 'proxies'" in str(e):
                # if the proxies parameter error is found, try different initialization methods
                import httpx
                client = OpenAI(
                    api_key=DEEPSEEK_API_KEY,
                    base_url="https://api.deepseek.com",
                    http_client=httpx.Client()
                )
            else:
                raise
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"},
            ],
            stream=False
        )
        
        if response.choices and response.choices[0].message.content:
            logger.info("DeepSeek API connection successful")
            return True
        else:
            logger.error("DeepSeek API returned an invalid response")
            return False
    except Exception as e:
        logger.exception(f"Error connecting to DeepSeek API: {e}")
        return False

def convert_text_to_paragraphs(text):
    """
    Convert text to paragraphs using DeepSeek API.
    
    Args:
        text (str): The text to convert
        
    Returns:
        str: The converted text
    """
    try:
        logger.debug(f"Converting text of length {len(text)}")
        
        # 修改初始化方式，避免 proxies 參數問題
        try:
            client = OpenAI(
                api_key=DEEPSEEK_API_KEY, 
                base_url="https://api.deepseek.com"
            )
        except TypeError as e:
            if "unexpected keyword argument 'proxies'" in str(e):
                # 如果發現 proxies 參數錯誤，嘗試不同的初始化方式
                import httpx
                client = OpenAI(
                    api_key=DEEPSEEK_API_KEY,
                    base_url="https://api.deepseek.com",
                    http_client=httpx.Client()
                )
            else:
                raise
        
        prompt = f"""請你將以下綜整為文意通順的段落文章，並具有標題。內容請務必都要包含進去，不要有任何省略，也不要擅自加入新內容。

{text}"""
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        
        result = response.choices[0].message.content
        logger.info(f"Successfully converted text, result length: {len(result)}")
        return result
    except Exception as e:
        logger.exception(f"Error converting text: {e}")
        return ""

def process_file(file_path):
    """
    Process a single file using DeepSeek API.
    
    Args:
        file_path (str): Path to the file to process
        
    Returns:
        str: The converted text
    """
    try:
        logger.debug(f"Processing file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Update current file in session state
        st.session_state.current_file = os.path.basename(file_path)
        
        # Convert the text
        converted_text = convert_text_to_paragraphs(content)
        
        logger.info(f"Processed file: {file_path}")
        return converted_text
    except Exception as e:
        logger.exception(f"Error processing file {file_path}: {e}")
        return ""

def save_converted_text(file_path, text):
    """
    Save the converted text to the original file.
    
    Args:
        file_path (str): Path to save the file
        text (str): Text to save
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.debug(f"Saving converted text to {file_path}")
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(text)
        
        st.session_state.converted_files.append(os.path.basename(file_path))
        logger.info(f"Saved converted text to {file_path}")
        return True
    except Exception as e:
        logger.exception(f"Error saving converted text to {file_path}: {e}")
        return False

def toggle_debug():
    """Toggle the debug mode"""
    st.session_state.streamlit_debug = not st.session_state.streamlit_debug
    logger.debug(f"Debug mode set to {st.session_state.streamlit_debug}")

def format_time(seconds):
    """Format seconds into a human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f} 秒"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds %= 60
        return f"{int(minutes)} 分 {int(seconds)} 秒"
    else:
        hours = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60
        return f"{int(hours)} 小時 {int(minutes)} 分 {int(seconds)} 秒"

# Main application UI
st.set_page_config(page_title="文本轉換工具", layout="wide")

st.title("文本段落轉換工具")

col1, col2 = st.columns([3, 1])

with col2:
    st.subheader("控制面板")
    
    # API connection test
    if st.button("測試 API 連接"):
        with st.spinner("正在測試 API 連接..."):
            st.session_state.api_connected = test_api_connection()
        
        if st.session_state.api_connected:
            st.success("API 連接成功！")
        else:
            st.error("API 連接失敗！請檢查 API 金鑰。")
    
    # Debug toggle
    if st.button("切換偵錯模式"):
        toggle_debug()
    
    # Show debug status
    st.write(f"偵錯模式: {'開啟' if st.session_state.streamlit_debug else '關閉'}")
    
    # Process button
    start_button = st.button("開始處理文件")
    
    # Progress information
    st.subheader("進度")
    
    # Display execution time
    if st.session_state.start_time:
        current_time = time.time()
        elapsed = current_time - st.session_state.start_time
        st.write(f"執行時間: {format_time(elapsed)}")
    
    # Display progress
    if st.session_state.converted_files:
        st.write(f"已處理: {len(st.session_state.converted_files)} 個檔案")
    
    if st.session_state.remaining_files:
        st.write(f"剩餘: {len(st.session_state.remaining_files)} 個檔案")
    
    # Current file
    if st.session_state.current_file:
        st.write(f"目前處理: {st.session_state.current_file}")

with col1:
    st.subheader("轉換結果")
    
    # Display the converted text
    if st.session_state.converted_text:
        st.text_area("轉換後的文本", st.session_state.converted_text, height=400)
        
        # Action buttons for the current conversion
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            if st.button("確認並保存"):
                if st.session_state.current_file and st.session_state.converted_text:
                    current_file_path = os.path.join(DATA_DIR, st.session_state.current_file)
                    if save_converted_text(current_file_path, st.session_state.converted_text):
                        st.success(f"已成功保存到 {current_file_path}")
                        
                        # Process next file if there are remaining files
                        if st.session_state.remaining_files:
                            next_file = st.session_state.remaining_files.pop(0)
                            st.session_state.converted_text = process_file(next_file)
                        else:
                            st.session_state.current_file = ""
                            st.session_state.converted_text = ""
                            st.success("所有文件處理完成！")
                            st.balloons()
                    else:
                        st.error("保存失敗，請查看偵錯日誌")
        
        with col_b:
            if st.button("取消"):
                if st.session_state.remaining_files:
                    next_file = st.session_state.remaining_files.pop(0)
                    st.session_state.converted_text = process_file(next_file)
                else:
                    st.session_state.current_file = ""
                    st.session_state.converted_text = ""
                    st.info("已取消處理")
        
        with col_c:
            if st.button("重新生成"):
                if st.session_state.current_file:
                    current_file_path = os.path.join(DATA_DIR, st.session_state.current_file)
                    st.session_state.converted_text = process_file(current_file_path)
                    st.info("已重新生成")

# Main process logic
if start_button:
    # Check API connection first
    if not st.session_state.api_connected:
        api_status = test_api_connection()
        st.session_state.api_connected = api_status
        
        if not api_status:
            st.error("API 連接失敗！請檢查 API 金鑰。")
            st.stop()
    
    # Get all text files
    all_files = get_all_text_files()
    
    if not all_files:
        st.error("找不到任何文本文件！")
        st.stop()
    
    # Initialize state
    st.session_state.converted_files = []
    st.session_state.remaining_files = all_files.copy()
    st.session_state.start_time = time.time()
    
    # Process the first file
    if st.session_state.remaining_files:
        first_file = st.session_state.remaining_files.pop(0)
        st.session_state.converted_text = process_file(first_file)
        st.experimental_rerun()

# Debug information in sidebar
if st.session_state.streamlit_debug:
    st.sidebar.subheader("偵錯資訊")
    
    st.sidebar.write("Session State:")
    st.sidebar.json(
        {
            "converted_files": st.session_state.converted_files,
            "current_file": st.session_state.current_file,
            "remaining_files": [os.path.basename(f) for f in st.session_state.remaining_files] if st.session_state.remaining_files else [],
            "api_connected": st.session_state.api_connected,
            "start_time": st.session_state.start_time,
        }
    )
    
    # Add a text area for viewing logs
    with open("debug.log", "r", encoding="utf-8") as log_file:
        try:
            logs = log_file.readlines()
            st.sidebar.text_area("日誌文件", "".join(logs[-20:]), height=200)
        except Exception as e:
            st.sidebar.error(f"無法讀取日誌文件: {e}") 