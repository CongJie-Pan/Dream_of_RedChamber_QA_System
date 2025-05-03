"""
Reasoning Module Streamlit Interface

This module provides a web interface for the LightRAG reasoning agent using Streamlit.
It allows users to ask complex questions and observe the reasoning process in detail,
visualizing the chain of thought and question decomposition.

Features:
- Interactive reasoning agent interface
- Chain of thought visualization
- Reasoning trace exploration
- Step-by-step process display
- Dependency graph visualization

Usage:
    streamlit run reasoning/streamlit_app.py

Author: CongJie Pan
Date: June 2024
"""

import streamlit as st
import os
import sys
import logging
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt
import networkx as nx
import io
import base64
from typing import Dict, List, Any, Optional, Union

# Add the parent directory to the path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import reasoning modules
from models import DeepSeekModel
from cot import ChainOfThought
from agent import ReasoningAgent
from pipeline import ReasoningPipeline
from visualization import ReasoningTraceVisualizer

# Configure logging
os.makedirs("logs/reasoning_streamlit", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/reasoning_streamlit/reasoning_streamlit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("reasoning_streamlit")

def initialize_reasoning_agent():
    """
    Initialize the reasoning agent and pipeline.
    
    Returns:
        tuple: (ReasoningAgent, ReasoningPipeline) instances
    """
    try:
        logger.info("Initializing DeepSeek model")
        model = DeepSeekModel()
        
        logger.info("Initializing reasoning agent")
        agent = ReasoningAgent(model=model)
        
        logger.info("Initializing reasoning pipeline")
        pipeline = ReasoningPipeline(reasoning_agent=agent)
        
        return agent, pipeline
    except Exception as e:
        logger.error(f"Error initializing reasoning components: {e}")
        st.error(f"初始化推理組件時出錯: {str(e)}")
        raise

def process_query(pipeline, query):
    """
    Process a user query through the reasoning pipeline.
    
    Args:
        pipeline: ReasoningPipeline instance
        query: User's question
        
    Returns:
        dict: Reasoning results including analysis, sub-questions, and visualization
    """
    try:
        logger.info(f"Processing query: {query}")
        
        # Process the query through the reasoning pipeline
        result = pipeline.process(query)
        
        logger.info(f"Query processed successfully, generated {len(result.get('sub_questions', []))} sub-questions")
        return result
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return {"error": str(e)}

def render_dependency_graph(sub_questions):
    """
    Generate and render a dependency graph for sub-questions.
    
    Args:
        sub_questions: List of sub-questions with dependencies
        
    Returns:
        str: Base64-encoded image of the dependency graph
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes for each sub-question
    for sq in sub_questions:
        sq_id = str(sq.get("id", 0))
        question = sq.get("question", f"Question {sq_id}")
        
        # Truncate long questions for display
        if len(question) > 40:
            question = question[:37] + "..."
        
        G.add_node(sq_id, label=question)
    
    # Add edges for dependencies
    for sq in sub_questions:
        sq_id = str(sq.get("id", 0))
        dependencies = sq.get("dependencies", [])
        
        for dep in dependencies:
            G.add_edge(str(dep), sq_id)
    
    # Generate the plot
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=3000, arrows=True)
    
    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    
    # Convert to base64 for embedding in HTML
    img_str = base64.b64encode(buf.read()).decode()
    return img_str

def display_reasoning_trace(trace_data):
    """
    Display the reasoning trace in the Streamlit UI.
    
    Args:
        trace_data: Reasoning trace data from the pipeline
    """
    # Display the original query
    st.subheader("原始問題")
    st.info(trace_data.get("query", "未提供問題"))
    
    # Display query analysis
    st.subheader("問題分析")
    analysis = trace_data.get("analysis", {})
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**複雜度:**", analysis.get("complexity", "未知"))
        st.write("**問題類型:**", analysis.get("question_type", "未知"))
    with col2:
        st.write("**需要分解:**", "是" if analysis.get("requires_decomposition", False) else "否")
        st.write("**領域:**", ", ".join(analysis.get("domains", ["未知"])))
    
    st.write("**關鍵概念:**", ", ".join(analysis.get("key_concepts", ["未知"])))
    
    # Display sub-questions if present
    if "sub_questions" in trace_data and trace_data["sub_questions"]:
        st.subheader("問題分解")
        
        # Display dependency graph if there are multiple sub-questions
        if len(trace_data["sub_questions"]) > 1:
            try:
                st.write("**依賴關係圖**")
                img_str = render_dependency_graph(trace_data["sub_questions"])
                st.image(f"data:image/png;base64,{img_str}", use_column_width=True)
            except Exception as e:
                logger.error(f"Error rendering dependency graph: {e}")
                st.error("無法生成依賴關係圖")
        
        # Display each sub-question with its details
        for i, sq in enumerate(trace_data["sub_questions"]):
            sq_id = sq.get("id", i+1)
            with st.expander(f"子問題 {sq_id}: {sq.get('question', '未知問題')}"):
                st.write("**相關性:**", sq.get("relevance", "未提供"))
                
                # Show dependencies if any
                if sq.get("dependencies"):
                    deps = [str(d) for d in sq.get("dependencies", [])]
                    st.write("**依賴於:**", ", ".join(deps))
                else:
                    st.write("**依賴於:** 無")
                
                # Show results for this sub-question if available
                if "sub_question_results" in trace_data and str(sq_id) in trace_data["sub_question_results"]:
                    result = trace_data["sub_question_results"][str(sq_id)]
                    st.write("**檢索結果:**")
                    st.json(result)
    
    # Display the final answer
    if "answer" in trace_data:
        st.subheader("最終答案")
        st.success(trace_data["answer"])
    
    # Display visualization if available
    if "visualization" in trace_data:
        st.subheader("推理過程可視化")
        
        # Display text visualization
        st.write("**文本可視化**")
        st.text(trace_data["visualization"].get("text", "無可視化數據"))
        
        # Display interactive visualization data if available
        if "interactive_data" in trace_data["visualization"]:
            st.write("**互動數據**")
            st.json(trace_data["visualization"]["interactive_data"])

def main():
    """
    Main function for the Streamlit app.
    """
    # Set page config
    st.set_page_config(
        page_title="推理代理可視化界面",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Page title
    st.title("🧠 推理代理可視化界面")
    
    # Sidebar for configuration
    with st.sidebar:
        st.title("配置")
        
        # Model selection (if multiple models supported)
        model_name = st.selectbox(
            "推理模型",
            ["DeepSeek R1"],
            index=0
        )
        
        # Visualization options
        st.subheader("可視化選項")
        show_dependency_graph = st.checkbox("顯示依賴關係圖", value=True)
        show_reasoning_trace = st.checkbox("顯示推理過程", value=True)
        
        # About section
        st.subheader("關於")
        st.markdown("""
        推理代理使用 DeepSeek R1 模型進行問題分解和推理。
        
        該界面允許您可視化複雜問題的分解過程，並探索推理鏈。
        """)
    
    # Initialize agent and pipeline if not already in session state
    if "agent" not in st.session_state or "pipeline" not in st.session_state:
        with st.spinner("初始化推理代理..."):
            try:
                agent, pipeline = initialize_reasoning_agent()
                st.session_state.agent = agent
                st.session_state.pipeline = pipeline
                st.success("推理代理初始化成功！")
            except Exception as e:
                st.error(f"初始化失敗: {e}")
                return
    
    # Display instructions
    st.markdown("""
    輸入一個複雜問題，系統會將其分解為較小的子問題，並顯示推理過程。
    
    範例問題:
    - 通過紅樓夢中的人物關係解釋"金陵十二釵"是誰？
    - 比較紅樓夢中王熙鳳和薛寶釵的性格特點和在故事中的作用。
    - 解釋紅樓夢中"葬花吟"的含義以及它如何反映林黛玉的心理狀態。
    """)
    
    # User input
    user_input = st.text_area("輸入您的問題:", height=100)
    
    # Process button
    if st.button("開始推理分析"):
        if not user_input:
            st.warning("請輸入問題再開始分析。")
        else:
            with st.spinner("正在分析問題..."):
                # Process the query
                result = process_query(st.session_state.pipeline, user_input)
                
                # Store result in session state
                st.session_state.last_result = result
                
                # Check for errors
                if "error" in result:
                    st.error(f"處理查詢時出錯: {result['error']}")
                else:
                    st.success("分析完成！")
    
    # Display results if available
    if "last_result" in st.session_state:
        st.header("推理結果")
        
        # Divider
        st.markdown("---")
        
        # Display the reasoning trace
        display_reasoning_trace(st.session_state.last_result)
        
        # Option to download results as JSON
        result_json = json.dumps(st.session_state.last_result, indent=2, ensure_ascii=False)
        st.download_button(
            label="下載完整結果 (JSON)",
            data=result_json,
            file_name=f"reasoning_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main() 