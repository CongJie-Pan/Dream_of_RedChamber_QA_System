"""
Reasoning Trace Visualization Module

This module provides visualization tools for displaying and analyzing reasoning traces.
It offers multiple visualization formats including HTML, text, and interactive visualizations.

Features:
- Text-based visualization of reasoning steps
- Rich HTML visualization with collapsible sections
- Interactive diagrams showing reasoning flow
- Timeline visualization of reasoning steps
- Support for exporting visualizations to various formats
"""

import json
import html
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import matplotlib.pyplot as plt
import networkx as nx
import base64
from io import BytesIO
from .config import logger


class ReasoningTraceVisualizer:
    """
    Class for visualizing reasoning traces in various formats.
    
    This class provides methods to visualize reasoning traces in different
    formats including text, HTML, and interactive diagrams.
    
    Attributes:
        cache_dir (str): Directory to cache visualization data
        last_trace (dict): The last trace data that was visualized
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the visualizer.
        
        Args:
            cache_dir (Optional[str]): Directory to cache visualization data
        """
        self.cache_dir = cache_dir
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            
        self.last_trace = None
    
    def generate_visualization(self, trace_data: Dict[str, Any], format_type: str = "text") -> str:
        """
        Generate a visualization of the reasoning trace in the specified format.
        
        Args:
            trace_data (Dict[str, Any]): The reasoning trace data
            format_type (str): The format to generate ('text', 'html', 'markdown', 'json')
            
        Returns:
            str: The visualization in the requested format
        """
        self.last_trace = trace_data
        
        if format_type.lower() == "text":
            return self._generate_text_visualization(trace_data)
        elif format_type.lower() == "html":
            return self._generate_html_visualization(trace_data)
        elif format_type.lower() == "markdown":
            return self._generate_markdown_visualization(trace_data)
        elif format_type.lower() == "json":
            return json.dumps(trace_data, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _generate_text_visualization(self, trace_data: Dict[str, Any]) -> str:
        """
        Generate a text-based visualization of the reasoning trace.
        
        Args:
            trace_data (Dict[str, Any]): The reasoning trace data
            
        Returns:
            str: Text-based visualization
        """
        lines = []
        
        # Add the original query
        lines.append("========== REASONING TRACE ==========")
        lines.append(f"Original Query: {trace_data.get('original_query', 'N/A')}")
        lines.append("")
        
        # Add the reasoning steps
        steps = trace_data.get("reasoning_steps", [])
        lines.append(f"Reasoning Process ({len(steps)} steps):")
        
        for i, step in enumerate(steps, 1):
            lines.append(f"Step {i}: {step}")
            
            # Find results for this step
            step_results = None
            if "sub_questions" in trace_data and i <= len(trace_data["sub_questions"]):
                sub_q = trace_data["sub_questions"][i-1]
                step_id = str(sub_q.get("id", i))
                
                if "results" in trace_data and step_id in trace_data["results"]:
                    step_results = trace_data["results"][step_id]
            
            # Add retrieval information if available
            if step_results:
                retrieval_source = step_results.get("source", "unknown")
                lines.append(f"  Source: {retrieval_source}")
                
                if "results" in step_results:
                    result_count = len(step_results["results"])
                    lines.append(f"  Retrieved {result_count} documents")
                
                if "error" in step_results:
                    lines.append(f"  Error: {step_results['error']}")
                
                lines.append("")
        
        # Add final answer if available
        if "answer" in trace_data:
            lines.append("\n=== Final Answer ===")
            lines.append(trace_data["answer"].get("answer", "No answer provided"))
        
        return "\n".join(lines)
    
    def _generate_html_visualization(self, trace_data: Dict[str, Any]) -> str:
        """
        Generate an HTML visualization of the reasoning trace.
        
        This creates a rich HTML representation with collapsible sections
        and formatted content.
        
        Args:
            trace_data (Dict[str, Any]): The reasoning trace data
            
        Returns:
            str: HTML visualization
        """
        # CSS styles for the visualization
        styles = """
        <style>
            .reasoning-trace {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                border: 1px solid #ddd;
                border-radius: 5px;
                background-color: #f9f9f9;
            }
            .header {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                border-radius: 5px 5px 0 0;
                margin-bottom: 15px;
                font-size: 1.2em;
            }
            .query {
                background-color: #f1f1f1;
                padding: 10px;
                border-left: 4px solid #4CAF50;
                margin-bottom: 20px;
                font-weight: bold;
            }
            .step {
                margin-bottom: 15px;
                border: 1px solid #ddd;
                border-radius: 5px;
                overflow: hidden;
            }
            .step-header {
                background-color: #f1f1f1;
                padding: 10px;
                cursor: pointer;
                font-weight: bold;
                border-bottom: 1px solid #ddd;
            }
            .step-content {
                padding: 10px;
                display: none;
                background-color: white;
            }
            .step-content.active {
                display: block;
            }
            .result {
                margin-top: 10px;
                padding: 10px;
                background-color: #f9f9f9;
                border-left: 3px solid #2196F3;
            }
            .error {
                color: red;
                font-weight: bold;
            }
            .final-answer {
                margin-top: 20px;
                padding: 15px;
                background-color: #e8f5e9;
                border-left: 5px solid #4CAF50;
                font-weight: bold;
            }
            .timeline {
                margin-top: 20px;
                padding: 10px;
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            .timeline-event {
                margin: 5px 0;
                padding: 5px;
                background-color: white;
                border-left: 3px solid #2196F3;
            }
            .toggle-all {
                margin-bottom: 10px;
                padding: 5px 10px;
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 3px;
                cursor: pointer;
            }
            .dependency-graph {
                margin: 20px 0;
                text-align: center;
            }
            .dependency-graph img {
                max-width: 100%;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
        </style>
        """
        
        # JavaScript for interactive elements
        script = """
        <script>
            function toggleContent(stepId) {
                var content = document.getElementById('content-' + stepId);
                if (content.style.display === 'block') {
                    content.style.display = 'none';
                } else {
                    content.style.display = 'block';
                }
            }
            
            function toggleAllSteps() {
                var contents = document.getElementsByClassName('step-content');
                var allHidden = true;
                
                // Check if any are visible
                for (var i = 0; i < contents.length; i++) {
                    if (contents[i].style.display === 'block') {
                        allHidden = false;
                        break;
                    }
                }
                
                // Toggle all based on current state
                for (var i = 0; i < contents.length; i++) {
                    contents[i].style.display = allHidden ? 'block' : 'none';
                }
                
                // Update button text
                document.getElementById('toggle-button').innerText = 
                    allHidden ? '收起所有步驟' : '展開所有步驟';
            }
        </script>
        """
        
        # Build HTML content
        html_content = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<meta charset='UTF-8'>",
            "<title>Reasoning Trace Visualization</title>",
            styles,
            "</head>",
            "<body>",
            "<div class='reasoning-trace'>",
            "<div class='header'>推理過程可視化</div>",  # Reasoning Process Visualization
            f"<div class='query'>問題: {html.escape(trace_data.get('original_query', 'N/A'))}</div>",  # Question
            "<button id='toggle-button' class='toggle-all' onclick='toggleAllSteps()'>展開所有步驟</button>",  # Expand all steps
            "<h3>推理步驟:</h3>"  # Reasoning Steps
        ]
        
        # Add reasoning steps
        steps = trace_data.get("reasoning_steps", [])
        for i, step in enumerate(steps, 1):
            html_content.append(f"<div class='step'>")
            html_content.append(f"<div class='step-header' onclick='toggleContent({i})'>步驟 {i}: {html.escape(step)}</div>")  # Step
            html_content.append(f"<div id='content-{i}' class='step-content'>")
            
            # Find results for this step
            step_results = None
            if "sub_questions" in trace_data and i <= len(trace_data["sub_questions"]):
                sub_q = trace_data["sub_questions"][i-1]
                step_id = str(sub_q.get("id", i))
                
                if "results" in trace_data and step_id in trace_data["results"]:
                    step_results = trace_data["results"][step_id]
            
            # Add retrieval information if available
            if step_results:
                retrieval_source = step_results.get("source", "unknown")
                html_content.append(f"<div><strong>來源:</strong> {html.escape(retrieval_source)}</div>")  # Source
                
                if "results" in step_results:
                    result_count = len(step_results["results"])
                    html_content.append(f"<div><strong>檢索到的文件數:</strong> {result_count}</div>")  # Retrieved documents
                    
                    # Display a sample of retrieved content if available
                    if result_count > 0 and isinstance(step_results["results"], list):
                        html_content.append("<div class='result'>")
                        html_content.append("<strong>檢索結果預覽:</strong><br>")  # Retrieved results preview
                        
                        # Show the first result or a sample
                        first_result = step_results["results"][0]
                        if isinstance(first_result, dict) and "content" in first_result:
                            content = first_result["content"]
                            # Limit content length for display
                            if len(content) > 300:
                                content = content[:300] + "..."
                            html_content.append(f"<div>{html.escape(content)}</div>")
                        
                        html_content.append("</div>")
                
                if "error" in step_results:
                    html_content.append(f"<div class='error'>錯誤: {html.escape(step_results['error'])}</div>")  # Error
            
            html_content.append("</div></div>")
        
        # Generate dependency graph for this trace
        if i == len(steps) and "sub_questions" in trace_data:
            try:
                graph_img = self._generate_dependency_graph(trace_data)
                if graph_img:
                    html_content.append("<div class='dependency-graph'>")
                    html_content.append("<h3>推理依賴關係圖:</h3>")  # Reasoning Dependency Graph
                    html_content.append(f"<img src='data:image/png;base64,{graph_img}' alt='依賴關係圖'>")  # Dependency Graph
                    html_content.append("</div>")
            except Exception as e:
                logger.error(f"Error generating dependency graph: {e}")
        
        # Add timeline visualization
        if "reasoning_timeline" in trace_data:
            timeline = trace_data.get("reasoning_timeline", [])
            if timeline:
                html_content.append("<div class='timeline'>")
                html_content.append("<h3>推理時間線:</h3>")  # Reasoning Timeline
                
                for event in timeline:
                    step_num = event.get("step_number", "?")
                    step_name = event.get("step_name", "Unknown")
                    elapsed = event.get("elapsed_time", 0)
                    html_content.append(f"<div class='timeline-event'>")
                    html_content.append(f"<strong>步驟 {step_num}:</strong> {html.escape(step_name)}")  # Step
                    html_content.append(f" <em>({elapsed:.2f}秒)</em>")  # seconds
                    html_content.append("</div>")
                
                html_content.append("</div>")
        
        # Add final answer if available
        if "answer" in trace_data:
            html_content.append("<div class='final-answer'>")
            html_content.append("<h3>最終答案:</h3>")  # Final Answer
            html_content.append(f"{html.escape(trace_data['answer'].get('answer', '未提供答案')).replace('\\n', '<br>').replace('\n', '<br>')}") 
            html_content.append("</div>")
        
        html_content.extend([
            "</div>",
            script,
            "</body>",
            "</html>"
        ])
        
        return "\n".join(html_content)
    
    def _generate_markdown_visualization(self, trace_data: Dict[str, Any]) -> str:
        """
        Generate a Markdown visualization of the reasoning trace.
        
        Args:
            trace_data (Dict[str, Any]): The reasoning trace data
            
        Returns:
            str: Markdown visualization
        """
        lines = []
        
        # Add the original query
        lines.append("# 推理過程可視化")  # Reasoning Process Visualization
        lines.append("")
        lines.append(f"**問題:** {trace_data.get('original_query', 'N/A')}")  # Question
        lines.append("")
        
        # Add the reasoning steps
        steps = trace_data.get("reasoning_steps", [])
        lines.append(f"## 推理步驟 ({len(steps)} 步)")  # Reasoning Steps
        lines.append("")
        
        for i, step in enumerate(steps, 1):
            lines.append(f"### 步驟 {i}: {step}")  # Step
            
            # Find results for this step
            step_results = None
            if "sub_questions" in trace_data and i <= len(trace_data["sub_questions"]):
                sub_q = trace_data["sub_questions"][i-1]
                step_id = str(sub_q.get("id", i))
                
                if "results" in trace_data and step_id in trace_data["results"]:
                    step_results = trace_data["results"][step_id]
            
            # Add retrieval information if available
            if step_results:
                retrieval_source = step_results.get("source", "unknown")
                lines.append(f"**來源:** {retrieval_source}")  # Source
                
                if "results" in step_results:
                    result_count = len(step_results["results"])
                    lines.append(f"**檢索到的文件數:** {result_count}")  # Retrieved documents
                
                if "error" in step_results:
                    lines.append(f"**錯誤:** {step_results['error']}")  # Error
                
                lines.append("")
        
        # Add timeline if available
        if "reasoning_timeline" in trace_data:
            timeline = trace_data.get("reasoning_timeline", [])
            if timeline:
                lines.append("## 推理時間線")  # Reasoning Timeline
                lines.append("")
                lines.append("| 步驟 | 階段 | 耗時(秒) |")  # Step | Stage | Time(s)
                lines.append("|------|------|---------|")
                
                for event in timeline:
                    step_num = event.get("step_number", "?")
                    step_name = event.get("step_name", "Unknown")
                    elapsed = event.get("elapsed_time", 0)
                    lines.append(f"| {step_num} | {step_name} | {elapsed:.2f} |")
                
                lines.append("")
        
        # Add final answer if available
        if "answer" in trace_data:
            lines.append("## 最終答案")  # Final Answer
            lines.append("")
            lines.append(trace_data["answer"].get("answer", "未提供答案"))  # No answer provided
        
        return "\n".join(lines)
    
    def _generate_dependency_graph(self, trace_data: Dict[str, Any]) -> Optional[str]:
        """
        Generate a dependency graph visualization of the reasoning steps.
        
        Args:
            trace_data (Dict[str, Any]): The reasoning trace data
            
        Returns:
            Optional[str]: Base64-encoded PNG image of the graph, or None if generation fails
        """
        try:
            # Import here to avoid dependency issues if matplotlib is not available
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            # Create a directed graph
            G = nx.DiGraph()
            
            # Add the original query as the root node
            query = trace_data.get("original_query", "Query")
            if len(query) > 40:
                query = query[:37] + "..."
            G.add_node("query", label=f"問題: {query}", shape="box", style="filled", fillcolor="lightblue")
            
            # Add nodes for each sub-question
            if "sub_questions" in trace_data:
                for i, sq in enumerate(trace_data["sub_questions"]):
                    sq_id = str(sq.get("id", i+1))
                    question = sq.get("question", f"Step {sq_id}")
                    if len(question) > 40:
                        question = question[:37] + "..."
                    
                    # Add the node
                    G.add_node(sq_id, label=f"步驟 {sq_id}: {question}", shape="box")
                    
                    # Connect to the query
                    G.add_edge("query", sq_id)
                    
                    # Add dependencies between steps
                    deps = sq.get("dependencies", [])
                    for dep in deps:
                        if str(dep) in G:
                            G.add_edge(str(dep), sq_id)
            
            # Add final answer node if available
            if "answer" in trace_data:
                answer = trace_data["answer"].get("answer", "")
                if answer:
                    if len(answer) > 50:
                        answer = answer[:47] + "..."
                    G.add_node("answer", label=f"回答: {answer}", shape="box", style="filled", fillcolor="lightgreen")
                    
                    # Connect all leaf nodes to the answer
                    leaf_nodes = [n for n, d in G.out_degree() if d == 0 and n != "answer"]
                    for node in leaf_nodes:
                        G.add_edge(node, "answer")
            
            # Create the plot
            plt.figure(figsize=(10, 8))
            pos = nx.spring_layout(G, seed=42)  # positions for all nodes
            
            # Draw nodes with labels
            nx.draw(G, pos, with_labels=False, node_color='lightblue', node_size=2000, alpha=0.8)
            
            # Draw node labels
            node_labels = nx.get_node_attributes(G, 'label')
            nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9, font_family='sans-serif')
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, arrows=True, arrowsize=15)
            
            # Save to a BytesIO object and encode as base64
            buf = BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=100)
            plt.close()
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            
            return img_str
        except Exception as e:
            logger.error(f"Failed to generate dependency graph: {e}")
            return None
    
    def generate_interactive_visualization(self, trace_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate data for an interactive visualization.
        
        This method prepares the data in a format suitable for interactive
        visualization libraries like D3.js or Plotly.
        
        Args:
            trace_data (Dict[str, Any]): The reasoning trace data
            
        Returns:
            Dict[str, Any]: Data for interactive visualization
        """
        # Prepare nodes and links for a graph visualization
        nodes = []
        links = []
        
        # Add the original query as the root node
        query = trace_data.get("original_query", "Query")
        nodes.append({
            "id": "query",
            "label": query[:50] + "..." if len(query) > 50 else query,
            "type": "query",
            "details": query
        })
        
        # Add nodes for each step/sub-question
        if "sub_questions" in trace_data:
            for i, sq in enumerate(trace_data["sub_questions"]):
                sq_id = str(sq.get("id", i+1))
                question = sq.get("question", f"Step {sq_id}")
                relevance = sq.get("relevance", "")
                
                # Find results for this step
                results_info = {}
                if "results" in trace_data and sq_id in trace_data["results"]:
                    results = trace_data["results"][sq_id]
                    results_info = {
                        "source": results.get("source", "unknown"),
                        "result_count": len(results.get("results", [])) if "results" in results else 0,
                        "error": results.get("error", None)
                    }
                
                # Add the node
                nodes.append({
                    "id": sq_id,
                    "label": f"步驟 {sq_id}: {question[:40]}..." if len(question) > 40 else f"步驟 {sq_id}: {question}",
                    "type": "step",
                    "details": {
                        "question": question,
                        "relevance": relevance,
                        "results": results_info
                    }
                })
                
                # Connect to the query
                links.append({
                    "source": "query",
                    "target": sq_id,
                    "type": "main"
                })
                
                # Add dependencies between steps
                deps = sq.get("dependencies", [])
                for dep in deps:
                    if str(dep) in [node["id"] for node in nodes]:
                        links.append({
                            "source": str(dep),
                            "target": sq_id,
                            "type": "dependency"
                        })
        
        # Add final answer if available
        if "answer" in trace_data:
            answer = trace_data["answer"].get("answer", "")
            if answer:
                answer_id = "answer"
                nodes.append({
                    "id": answer_id,
                    "label": "最終答案",  # Final Answer
                    "type": "answer",
                    "details": {
                        "text": answer
                    }
                })
                
                # Connect leaf nodes to the answer
                leaf_nodes = [node["id"] for node in nodes if node["id"] != answer_id and 
                             node["id"] not in [link["target"] for link in links if link["type"] == "dependency"]]
                
                for node_id in leaf_nodes:
                    if node_id != "query":  # Don't connect query directly to answer
                        links.append({
                            "source": node_id,
                            "target": answer_id,
                            "type": "result"
                        })
        
        # Add timeline data if available
        timeline = None
        if "reasoning_timeline" in trace_data:
            timeline = trace_data.get("reasoning_timeline", [])
        
        # Return the complete visualization data
        return {
            "nodes": nodes,
            "links": links,
            "timeline": timeline,
            "query": trace_data.get("original_query", ""),
            "timestamp": datetime.now().isoformat()
        }
    
    def save_visualization(self, format_type: str = "html", filename: Optional[str] = None) -> str:
        """
        Save the last generated visualization to a file.
        
        Args:
            format_type (str): The format to save ('html', 'markdown', 'json')
            filename (Optional[str]): The filename to save to. If None, a default name is used.
            
        Returns:
            str: Path to the saved file
            
        Raises:
            ValueError: If no visualization has been generated yet
        """
        if not self.last_trace:
            raise ValueError("No visualization has been generated yet")
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reasoning_trace_{timestamp}.{format_type}"
        
        if self.cache_dir:
            filepath = os.path.join(self.cache_dir, filename)
        else:
            filepath = filename
        
        # Generate the visualization
        content = self.generate_visualization(self.last_trace, format_type)
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return filepath
    
    def generate_timeline_chart(self, trace_data: Dict[str, Any]) -> Optional[str]:
        """
        Generate a timeline chart of the reasoning process.
        
        Args:
            trace_data (Dict[str, Any]): The reasoning trace data
            
        Returns:
            Optional[str]: Base64-encoded PNG image of the timeline chart
        """
        if "reasoning_timeline" not in trace_data:
            return None
        
        timeline = trace_data.get("reasoning_timeline", [])
        if not timeline:
            return None
        
        try:
            # Create the timeline plot
            plt.figure(figsize=(10, 6))
            
            steps = []
            times = []
            
            for event in timeline:
                step_name = event.get("step_name", "Unknown")
                if len(step_name) > 20:
                    step_name = step_name[:17] + "..."
                steps.append(f"{event.get('step_number', '?')}. {step_name}")
                times.append(event.get("elapsed_time", 0))
            
            # Reverse order for bottom-to-top display
            steps.reverse()
            times.reverse()
            
            # Create horizontal bar chart
            plt.barh(steps, times, color='skyblue')
            plt.xlabel('時間 (秒)')  # Time (seconds)
            plt.ylabel('步驟')  # Step
            plt.title('推理過程耗時')  # Reasoning Process Timeline
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save to a BytesIO object and encode as base64
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            plt.close()
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            
            return img_str
        except Exception as e:
            logger.error(f"Failed to generate timeline chart: {e}")
            return None
    
    def export_visualization(self, trace_data: Dict[str, Any], export_format: str = "html", 
                           output_dir: Optional[str] = None) -> str:
        """
        Export the visualization to a file in the specified format.
        
        Args:
            trace_data (Dict[str, Any]): The reasoning trace data
            export_format (str): Format to export ('html', 'json', 'markdown', 'pdf')
            output_dir (Optional[str]): Directory to save the export
            
        Returns:
            str: Path to the exported file
        """
        if not output_dir:
            output_dir = self.cache_dir or "."
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reasoning_trace_{timestamp}.{export_format}"
        filepath = os.path.join(output_dir, filename)
        
        if export_format == 'json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(trace_data, f, indent=2)
        elif export_format == 'html':
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self._generate_html_visualization(trace_data))
        elif export_format == 'markdown' or export_format == 'md':
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self._generate_markdown_visualization(trace_data))
        elif export_format == 'pdf':
            try:
                # This requires additional dependencies like wkhtmltopdf
                import pdfkit
                html_content = self._generate_html_visualization(trace_data)
                pdfkit.from_string(html_content, filepath)
            except ImportError:
                logger.error("pdfkit module not found. Install with 'pip install pdfkit wkhtmltopdf'")
                return "Error: PDF export requires additional dependencies"
            except Exception as e:
                logger.error(f"Error generating PDF: {e}")
                return f"Error generating PDF: {e}"
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
            
        return filepath 