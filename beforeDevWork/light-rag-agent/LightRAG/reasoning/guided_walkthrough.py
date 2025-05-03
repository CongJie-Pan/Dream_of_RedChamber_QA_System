"""
Guided Walkthrough for Complex Reasoning Traces

This module provides interactive guided walkthrough functionality
for exploring and understanding complex reasoning traces.

Features:
- Step-by-step explanations of reasoning processes
- Interactive exploration of reasoning chains
- Simplified and detailed views of reasoning steps
- Educational content about the reasoning approach
"""

import json
from typing import Dict, List, Any, Optional, Union, Callable
from .config import logger


class GuidedWalkthrough:
    """
    Interactive guided walkthrough for reasoning traces.
    
    This class provides functionality for exploring complex reasoning
    traces in an educational and guided manner, breaking down the reasoning
    process into understandable steps.
    
    Attributes:
        current_trace (Dict[str, Any]): The current reasoning trace data
        explanations (Dict[str, str]): Explanations for different reasoning steps
        current_step (int): The current step in the walkthrough
    """
    
    def __init__(self):
        """Initialize the guided walkthrough."""
        self.current_trace = None
        self.explanations = {}
        self.current_step = 0
        self.total_steps = 0
    
    def load_trace(self, trace_data: Dict[str, Any]) -> None:
        """
        Load a reasoning trace for walkthrough.
        
        Args:
            trace_data (Dict[str, Any]): The reasoning trace data
        """
        self.current_trace = trace_data
        self.current_step = 0
        
        if "reasoning_steps" in trace_data:
            self.total_steps = len(trace_data["reasoning_steps"])
            # Generate explanations for each step
            self._generate_explanations()
        else:
            self.total_steps = 0
            logger.warning("No reasoning steps found in the trace data")
    
    def _generate_explanations(self) -> None:
        """Generate explanations for each reasoning step."""
        self.explanations = {}
        
        if not self.current_trace or "reasoning_steps" not in self.current_trace:
            return
        
        steps = self.current_trace.get("reasoning_steps", [])
        
        # Generate generic explanations for each step
        for i, step in enumerate(steps):
            step_id = str(i + 1)
            
            # Find corresponding sub-question data
            sub_q = None
            if "sub_questions" in self.current_trace and i < len(self.current_trace["sub_questions"]):
                sub_q = self.current_trace["sub_questions"][i]
            
            # Find results for this step
            step_results = None
            if sub_q:
                sub_q_id = str(sub_q.get("id", i+1))
                if "results" in self.current_trace and sub_q_id in self.current_trace["results"]:
                    step_results = self.current_trace["results"][sub_q_id]
            
            # Generate the explanation
            explanation = self._create_step_explanation(i+1, step, sub_q, step_results)
            self.explanations[step_id] = explanation
    
    def _create_step_explanation(self, step_num: int, step_text: str, 
                               sub_question: Optional[Dict[str, Any]] = None,
                               step_results: Optional[Dict[str, Any]] = None) -> str:
        """
        Create an explanation for a single reasoning step.
        
        Args:
            step_num (int): The step number
            step_text (str): The text of the step
            sub_question (Optional[Dict[str, Any]]): The sub-question data if available
            step_results (Optional[Dict[str, Any]]): The results for this step if available
            
        Returns:
            str: The step explanation
        """
        explanation = []
        
        # Add the step title and text
        explanation.append(f"### 步驟 {step_num}: {step_text}")
        explanation.append("")
        
        # Add relevance info if available
        if sub_question and "relevance" in sub_question:
            relevance = sub_question["relevance"]
            explanation.append(f"**為什麼這個步驟很重要**: {relevance}")
            explanation.append("")
        
        # Add dependencies info if available
        if sub_question and "dependencies" in sub_question:
            deps = sub_question["dependencies"]
            if deps:
                deps_str = ", ".join([f"步驟 {d}" for d in deps])
                explanation.append(f"**依賴於前面的步驟**: {deps_str}")
                explanation.append("")
        
        # Add retrieval strategy if available
        if step_results:
            retrieval_source = step_results.get("source", "unknown")
            explanation.append(f"**檢索策略**: {retrieval_source}")
            
            if "parameters" in step_results:
                params = step_results["parameters"]
                explanation.append("**檢索參數**:")
                for param, value in params.items():
                    explanation.append(f"- {param}: {value}")
            
            explanation.append("")
            
            # Add retrieval results summary if available
            if "results" in step_results:
                results = step_results["results"]
                result_count = len(results)
                explanation.append(f"**檢索結果**: 共找到 {result_count} 個文檔片段")
                
                if result_count > 0:
                    explanation.append("**關鍵資訊摘要**:")
                    for i, result in enumerate(results[:3]):  # Show up to 3 results
                        if isinstance(result, dict) and "content" in result:
                            content = result["content"]
                            preview = content[:100] + "..." if len(content) > 100 else content
                            explanation.append(f"- 文檔 {i+1}: {preview}")
                
                explanation.append("")
        
        # Add educational context about reasoning approach
        explanation.append("**這個步驟在推理過程中的作用**:")
        
        # Check if this is the first step
        if step_num == 1:
            explanation.append("這是推理過程的起點，設定問題解決的初始方向。")
        # Check if this is the last step
        elif step_num == self.total_steps:
            explanation.append("這是推理鏈的最後一步，向最終答案過渡。")
        # Middle steps
        else:
            explanation.append("這是推理鏈中的中間步驟，構建從問題到答案的橋樑。")
        
        explanation.append("")
        
        return "\n".join(explanation)
    
    def start_walkthrough(self) -> Dict[str, Any]:
        """
        Start the guided walkthrough from the beginning.
        
        Returns:
            Dict[str, Any]: Information about the first step
        """
        self.current_step = 0
        return self.next_step()
    
    def next_step(self) -> Dict[str, Any]:
        """
        Move to the next step in the walkthrough.
        
        Returns:
            Dict[str, Any]: Information about the next step
            or a completion message if the walkthrough is complete
        """
        if not self.current_trace:
            return {"error": "No trace loaded. Please load a trace first."}
        
        if self.current_step >= self.total_steps:
            # We're already at the end
            return {
                "complete": True,
                "message": "推理過程導覽已完成。",  # Reasoning walkthrough completed
                "original_query": self.current_trace.get("original_query", ""),
                "final_answer": self.current_trace.get("answer", {}).get("answer", "未提供答案")  # No answer provided
            }
        
        # Increment the step
        self.current_step += 1
        step_id = str(self.current_step)
        
        # Get the step text
        step_text = ""
        if "reasoning_steps" in self.current_trace and self.current_step <= len(self.current_trace["reasoning_steps"]):
            step_text = self.current_trace["reasoning_steps"][self.current_step - 1]
        
        # Get the explanation
        explanation = self.explanations.get(step_id, "無可用解釋。")  # No explanation available
        
        # Get progress information
        progress = {
            "current": self.current_step,
            "total": self.total_steps,
            "percent": round(self.current_step / self.total_steps * 100) if self.total_steps > 0 else 0
        }
        
        # Create the step information
        step_info = {
            "step_number": self.current_step,
            "step_text": step_text,
            "explanation": explanation,
            "progress": progress,
            "is_first": self.current_step == 1,
            "is_last": self.current_step == self.total_steps
        }
        
        return step_info
    
    def previous_step(self) -> Dict[str, Any]:
        """
        Move to the previous step in the walkthrough.
        
        Returns:
            Dict[str, Any]: Information about the previous step
            or an error if already at the beginning
        """
        if not self.current_trace:
            return {"error": "No trace loaded. Please load a trace first."}
        
        if self.current_step <= 1:
            # We're already at the beginning
            return {"error": "Already at the first step."}
        
        # Decrement the step
        self.current_step -= 1
        step_id = str(self.current_step)
        
        # Get the step text
        step_text = ""
        if "reasoning_steps" in self.current_trace and self.current_step <= len(self.current_trace["reasoning_steps"]):
            step_text = self.current_trace["reasoning_steps"][self.current_step - 1]
        
        # Get the explanation
        explanation = self.explanations.get(step_id, "無可用解釋。")  # No explanation available
        
        # Get progress information
        progress = {
            "current": self.current_step,
            "total": self.total_steps,
            "percent": round(self.current_step / self.total_steps * 100) if self.total_steps > 0 else 0
        }
        
        # Create the step information
        step_info = {
            "step_number": self.current_step,
            "step_text": step_text,
            "explanation": explanation,
            "progress": progress,
            "is_first": self.current_step == 1,
            "is_last": self.current_step == self.total_steps
        }
        
        return step_info
    
    def jump_to_step(self, step_num: int) -> Dict[str, Any]:
        """
        Jump to a specific step in the walkthrough.
        
        Args:
            step_num (int): The step number to jump to
            
        Returns:
            Dict[str, Any]: Information about the specified step
            or an error if the step number is invalid
        """
        if not self.current_trace:
            return {"error": "No trace loaded. Please load a trace first."}
        
        if step_num < 1 or step_num > self.total_steps:
            return {"error": f"Invalid step number. Must be between 1 and {self.total_steps}."}
        
        # Set the current step
        self.current_step = step_num
        step_id = str(self.current_step)
        
        # Get the step text
        step_text = ""
        if "reasoning_steps" in self.current_trace and self.current_step <= len(self.current_trace["reasoning_steps"]):
            step_text = self.current_trace["reasoning_steps"][self.current_step - 1]
        
        # Get the explanation
        explanation = self.explanations.get(step_id, "無可用解釋。")  # No explanation available
        
        # Get progress information
        progress = {
            "current": self.current_step,
            "total": self.total_steps,
            "percent": round(self.current_step / self.total_steps * 100) if self.total_steps > 0 else 0
        }
        
        # Create the step information
        step_info = {
            "step_number": self.current_step,
            "step_text": step_text,
            "explanation": explanation,
            "progress": progress,
            "is_first": self.current_step == 1,
            "is_last": self.current_step == self.total_steps
        }
        
        return step_info
    
    def get_overview(self) -> Dict[str, Any]:
        """
        Get an overview of the entire reasoning trace.
        
        Returns:
            Dict[str, Any]: Overview of the reasoning trace
            including the query, steps, and answer
        """
        if not self.current_trace:
            return {"error": "No trace loaded. Please load a trace first."}
        
        # Create the overview
        overview = {
            "original_query": self.current_trace.get("original_query", ""),
            "total_steps": self.total_steps,
            "steps": [],
            "final_answer": self.current_trace.get("answer", {}).get("answer", "未提供答案")  # No answer provided
        }
        
        # Add step summaries
        if "reasoning_steps" in self.current_trace:
            for i, step in enumerate(self.current_trace["reasoning_steps"]):
                # Create a brief summary for each step
                step_summary = {
                    "step_number": i + 1,
                    "step_text": step
                }
                overview["steps"].append(step_summary)
        
        return overview
    
    def get_step_details(self, step_num: int) -> Dict[str, Any]:
        """
        Get detailed information about a specific step.
        
        Args:
            step_num (int): The step number
            
        Returns:
            Dict[str, Any]: Detailed information about the step
            or an error if the step number is invalid
        """
        if not self.current_trace:
            return {"error": "No trace loaded. Please load a trace first."}
        
        if step_num < 1 or step_num > self.total_steps:
            return {"error": f"Invalid step number. Must be between 1 and {self.total_steps}."}
        
        step_id = str(step_num)
        
        # Get the step text
        step_text = ""
        if "reasoning_steps" in self.current_trace and step_num <= len(self.current_trace["reasoning_steps"]):
            step_text = self.current_trace["reasoning_steps"][step_num - 1]
        
        # Find corresponding sub-question data
        sub_q = None
        if "sub_questions" in self.current_trace and step_num <= len(self.current_trace["sub_questions"]):
            sub_q = self.current_trace["sub_questions"][step_num - 1]
        
        # Find results for this step
        step_results = None
        if sub_q:
            sub_q_id = str(sub_q.get("id", step_num))
            if "results" in self.current_trace and sub_q_id in self.current_trace["results"]:
                step_results = self.current_trace["results"][sub_q_id]
        
        # Get dependencies
        dependencies = []
        if sub_q and "dependencies" in sub_q:
            for dep in sub_q["dependencies"]:
                dep_step = next((i+1 for i, sq in enumerate(self.current_trace.get("sub_questions", []))
                              if sq.get("id") == dep), None)
                if dep_step:
                    dep_text = self.current_trace["reasoning_steps"][dep_step - 1]
                    dependencies.append({
                        "step_number": dep_step,
                        "step_text": dep_text
                    })
        
        # Get dependent steps
        dependent_steps = []
        for i, sq in enumerate(self.current_trace.get("sub_questions", [])):
            if "dependencies" in sq and sub_q and sub_q.get("id") in sq["dependencies"]:
                step_num = i + 1
                step_text = self.current_trace["reasoning_steps"][i]
                dependent_steps.append({
                    "step_number": step_num,
                    "step_text": step_text
                })
        
        # Create detailed information
        details = {
            "step_number": step_num,
            "step_text": step_text,
            "explanation": self.explanations.get(step_id, "無可用解釋。"),  # No explanation available
            "relevance": sub_q.get("relevance", "") if sub_q else "",
            "dependencies": dependencies,
            "dependent_steps": dependent_steps,
            "retrieval_source": step_results.get("source", "unknown") if step_results else None,
            "retrieval_parameters": step_results.get("parameters", {}) if step_results else None,
            "result_count": len(step_results.get("results", [])) if step_results and "results" in step_results else 0
        }
        
        return details
    
    def generate_html_walkthrough(self) -> str:
        """
        Generate an HTML version of the guided walkthrough.
        
        Returns:
            str: HTML representation of the walkthrough
            or an error message if no trace is loaded
        """
        if not self.current_trace:
            return "<p>Error: No trace loaded. Please load a trace first.</p>"
        
        # Generate the HTML
        html = []
        
        # Add the title and CSS
        html.append("""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>推理過程導覽</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }
                h1, h2, h3 {
                    color: #2c3e50;
                }
                .query {
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-left: 4px solid #3498db;
                    margin-bottom: 20px;
                }
                .step {
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    margin-bottom: 20px;
                    overflow: hidden;
                }
                .step-header {
                    background-color: #f1f8ff;
                    padding: 10px 15px;
                    border-bottom: 1px solid #ddd;
                    font-weight: bold;
                }
                .step-content {
                    padding: 15px;
                }
                .step-explanation {
                    background-color: #f9f9f9;
                    padding: 10px;
                    border-left: 3px solid #28a745;
                    margin-top: 10px;
                }
                .answer {
                    background-color: #e8f5e9;
                    padding: 15px;
                    border-left: 4px solid #28a745;
                    margin-top: 30px;
                }
                .progress-bar {
                    background-color: #e9ecef;
                    border-radius: 0.25rem;
                    margin: 20px 0;
                }
                .progress-bar-fill {
                    background-color: #007bff;
                    height: 10px;
                    border-radius: 0.25rem;
                }
            </style>
        </head>
        <body>
        """)
        
        # Add the query
        html.append(f'<h1>推理過程導覽</h1>')
        html.append(f'<div class="query"><strong>問題:</strong> {self.current_trace.get("original_query", "")}</div>')
        
        # Add the progress bar
        html.append('<div class="progress-bar"><div class="progress-bar-fill" style="width: 100%;"></div></div>')
        
        # Add overview
        html.append('<h2>推理步驟總覽</h2>')
        html.append('<ol>')
        if "reasoning_steps" in self.current_trace:
            for step in self.current_trace["reasoning_steps"]:
                html.append(f'<li>{step}</li>')
        html.append('</ol>')
        
        # Add each step
        html.append('<h2>詳細步驟說明</h2>')
        
        for i in range(1, self.total_steps + 1):
            step_id = str(i)
            
            # Get the step text
            step_text = ""
            if "reasoning_steps" in self.current_trace and i <= len(self.current_trace["reasoning_steps"]):
                step_text = self.current_trace["reasoning_steps"][i - 1]
            
            # Get the explanation
            explanation = self.explanations.get(step_id, "")
            
            # Add the step
            html.append(f'<div class="step" id="step-{i}">')
            html.append(f'<div class="step-header">步驟 {i}: {step_text}</div>')
            html.append('<div class="step-content">')
            html.append('<div class="step-explanation">')
            html.append(explanation.replace('\n', '<br/>'))
            html.append('</div>')
            html.append('</div>')
            html.append('</div>')
        
        # Add the final answer
        answer = self.current_trace.get("answer", {}).get("answer", "未提供答案")
        html.append(f'<div class="answer"><h3>最終答案:</h3>{answer}</div>')
        
        # Add the closing tags
        html.append('</body>')
        html.append('</html>')
        
        return "\n".join(html)


class ReasoningEducator:
    """
    Educational component for explaining reasoning approaches.
    
    This class provides educational content about reasoning approaches,
    explaining concepts, techniques, and patterns used in the reasoning process.
    
    Attributes:
        glossary (Dict[str, str]): Glossary of reasoning terms
        patterns (Dict[str, str]): Explanation of common reasoning patterns
    """
    
    def __init__(self):
        """Initialize the reasoning educator with educational content."""
        # Glossary of reasoning terms
        self.glossary = {
            "chain_of_thought": "思考鏈 (Chain of Thought, CoT) 是一種推理技術，讓AI通過明確的推理步驟來解決問題，類似於人類的逐步思考過程。",
            "decomposition": "問題分解是將複雜問題分成更小、更易管理的子問題的策略。",
            "integration": "結果整合是將多個子問題的答案組合成一個連貫完整的最終答案的過程。",
            "dependency": "依賴關係指的是當一個推理步驟需要先解決另一個步驟才能進行時的關係。",
            "retrieval": "檢索是從知識庫或外部資源中尋找與問題相關的信息的過程。",
            "verification": "驗證是檢查推理步驟和結論的正確性和一致性的過程。"
        }
        
        # Common reasoning patterns
        self.patterns = {
            "sequential": "順序推理模式按照確定的順序解決一系列相關問題，每個問題都建立在前一個問題的結果之上。",
            "divergent": "發散推理模式從一個核心問題開始，向外擴展到多個相關的子問題進行探索。",
            "convergent": "聚合推理模式從多個角度或來源收集證據，然後將它們整合為一個答案。",
            "recursive": "遞歸推理模式將問題分解成相似但規模更小的子問題，直到達到可以直接解決的基本情況。",
            "comparative": "比較推理模式通過對比不同選項、概念或觀點來得出結論。"
        }
    
    def get_term_definition(self, term: str) -> str:
        """
        Get the definition of a reasoning term.
        
        Args:
            term (str): The term to define
            
        Returns:
            str: The definition of the term or a message if not found
        """
        # Normalize the term
        normalized_term = term.lower().replace(" ", "_")
        
        return self.glossary.get(normalized_term, "未找到此術語的定義。")  # Definition not found
    
    def get_pattern_explanation(self, pattern: str) -> str:
        """
        Get the explanation of a reasoning pattern.
        
        Args:
            pattern (str): The pattern to explain
            
        Returns:
            str: The explanation of the pattern or a message if not found
        """
        # Normalize the pattern
        normalized_pattern = pattern.lower().replace(" ", "_")
        
        return self.patterns.get(normalized_pattern, "未找到此推理模式的解釋。")  # Explanation not found
    
    def identify_pattern(self, trace_data: Dict[str, Any]) -> str:
        """
        Identify the reasoning pattern used in a trace.
        
        Args:
            trace_data (Dict[str, Any]): The reasoning trace data
            
        Returns:
            str: The identified reasoning pattern
        """
        # Check for dependencies to determine the pattern
        if "sub_questions" not in trace_data:
            return "unknown"
        
        sub_questions = trace_data["sub_questions"]
        
        # Check if there are any dependencies
        has_dependencies = any("dependencies" in sq and sq["dependencies"] for sq in sub_questions)
        
        if not has_dependencies:
            # No dependencies, likely sequential or divergent
            return "sequential"
        
        # Count dependency types
        dep_count = 0
        for sq in sub_questions:
            if "dependencies" in sq:
                dep_count += len(sq["dependencies"])
        
        # Determine pattern based on dependency structure
        if dep_count > len(sub_questions) * 0.7:
            # High dependency ratio, likely recursive
            return "recursive"
        elif dep_count > len(sub_questions) * 0.4:
            # Medium dependency ratio, likely convergent
            return "convergent"
        else:
            # Low dependency ratio, likely comparative
            return "comparative"
    
    def generate_educational_content(self, trace_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate educational content about the reasoning in a trace.
        
        Args:
            trace_data (Dict[str, Any]): The reasoning trace data
            
        Returns:
            Dict[str, str]: Educational content about the reasoning
        """
        # Identify the pattern
        pattern = self.identify_pattern(trace_data)
        
        # Generate content
        content = {
            "identified_pattern": pattern,
            "pattern_explanation": self.get_pattern_explanation(pattern),
            "key_concepts": [],
            "tips": []
        }
        
        # Add key concepts
        content["key_concepts"].append({
            "term": "chain_of_thought",
            "definition": self.get_term_definition("chain_of_thought")
        })
        
        content["key_concepts"].append({
            "term": "decomposition",
            "definition": self.get_term_definition("decomposition")
        })
        
        content["key_concepts"].append({
            "term": "integration",
            "definition": self.get_term_definition("integration")
        })
        
        # Add tips based on pattern
        if pattern == "sequential":
            content["tips"].append("留意每個步驟如何建立在前一個步驟之上。")
            content["tips"].append("注意信息是如何從早期步驟流向後續步驟的。")
        elif pattern == "convergent":
            content["tips"].append("觀察不同子問題的答案如何聚合成一個統一的結論。")
            content["tips"].append("注意可能出現的相互矛盾的證據以及模型如何解決這些矛盾。")
        elif pattern == "recursive":
            content["tips"].append("識別問題是如何分解成相似但規模更小的子問題的。")
            content["tips"].append("尋找基本情況——解決方案不再需要進一步分解的點。")
        
        return content


def create_walkthrough_from_file(file_path: str) -> GuidedWalkthrough:
    """
    Create a guided walkthrough from a trace file.
    
    Args:
        file_path (str): Path to the trace JSON file
        
    Returns:
        GuidedWalkthrough: The initialized walkthrough
        
    Raises:
        FileNotFoundError: If the file is not found
        json.JSONDecodeError: If the file is not valid JSON
    """
    try:
        # Read the trace file
        with open(file_path, "r", encoding="utf-8") as f:
            trace_data = json.load(f)
        
        # Create and initialize the walkthrough
        walkthrough = GuidedWalkthrough()
        walkthrough.load_trace(trace_data)
        
        return walkthrough
    except FileNotFoundError:
        logger.error(f"Trace file not found: {file_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in trace file: {file_path}")
        raise 