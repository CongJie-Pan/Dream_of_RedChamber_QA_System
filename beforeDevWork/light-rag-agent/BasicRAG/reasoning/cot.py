from typing import Dict, List, Optional, Tuple
from .models import DeepSeekModel
import json

class ChainOfThought:
    """
    Chain of Thought (COT) reasoning module.
    
    This class implements Chain of Thought reasoning methodology to break down
    complex questions into sub-questions and manage the reasoning process.
    
    Attributes:
        model (DeepSeekModel): The language model used for reasoning.
        max_sub_questions (int): Maximum number of sub-questions to generate.
    """
    
    def __init__(self, model: Optional[DeepSeekModel] = None, max_sub_questions: int = 5):
        """
        Initialize the Chain of Thought reasoning module.
        
        Args:
            model (Optional[DeepSeekModel]): The language model for reasoning. If None, creates a new instance.
            max_sub_questions (int): Maximum number of sub-questions to generate.
        """
        self.model = model or DeepSeekModel()
        self.max_sub_questions = max_sub_questions
    
    def decompose_question(self, query: str) -> List[Dict]:
        """
        Decompose a complex question into sub-questions using Chain of Thought reasoning.
        
        Args:
            query (str): The complex query to decompose.
            
        Returns:
            List[Dict]: List of sub-questions with metadata.
        """
        prompt = f"""
        I need to answer this complex question: "{query}"
        
        Please help me break this down into smaller, focused sub-questions that will help me answer the original question.
        For each sub-question:
        1. Formulate a clear, specific question.
        2. Explain why this sub-question is relevant to the original question.
        3. Suggest what kind of information would help answer this sub-question.
        4. Indicate which other sub-questions (if any) depend on the answer to this one.
        
        Please limit your response to {self.max_sub_questions} sub-questions at most.
        Format your response as:
        
        Sub-question 1: [The question]
        Relevance: [Why it's relevant]
        Information needed: [What kind of information would help]
        Dependencies: [None or list of dependent questions by number]
        
        Sub-question 2: ...
        """
        
        response = self.model.call(prompt, {"temperature": 0.3})
        
        # Parse the response to extract structured sub-questions
        return self._parse_sub_questions(response)
    
    def _parse_sub_questions(self, response: str) -> List[Dict]:
        """
        Parse the model's response to extract structured sub-questions.
        
        Args:
            response (str): Raw model response containing sub-questions.
            
        Returns:
            List[Dict]: List of parsed sub-questions with metadata.
        """
        sub_questions = []
        current_question = {}
        
        # Simple parsing logic - this could be improved with regex or more robust parsing
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("Sub-question"):
                if current_question and "question" in current_question:
                    sub_questions.append(current_question)
                current_question = {"id": len(sub_questions) + 1}
                # Extract the question text
                question_parts = line.split(":", 1)
                if len(question_parts) > 1:
                    current_question["question"] = question_parts[1].strip()
                else:
                    current_question["question"] = ""
            elif line.startswith("Relevance:") and current_question:
                current_question["relevance"] = line.split(":", 1)[1].strip()
            elif line.startswith("Information needed:") and current_question:
                current_question["info_needed"] = line.split(":", 1)[1].strip()
            elif line.startswith("Dependencies:") and current_question:
                deps = line.split(":", 1)[1].strip()
                if deps.lower() == "none":
                    current_question["dependencies"] = []
                else:
                    # Parse dependencies as list of question numbers
                    try:
                        deps_list = [int(d.strip()) for d in deps.replace("[", "").replace("]", "").split(",") if d.strip().isdigit()]
                        current_question["dependencies"] = deps_list
                    except:
                        current_question["dependencies"] = []
        
        # Add the last question if it exists
        if current_question and "question" in current_question:
            sub_questions.append(current_question)
            
        return sub_questions
    
    def create_retrieval_sequence(self, sub_questions: List[Dict]) -> List[int]:
        """
        Create an optimal sequence for processing sub-questions based on dependencies.
        
        Args:
            sub_questions (List[Dict]): List of sub-questions with dependency information.
            
        Returns:
            List[int]: Ordered list of sub-question IDs for processing.
        """
        # Create a simple topological sort based on dependencies
        processed = set()
        sequence = []
        
        def process_question(question_id):
            if question_id in processed:
                return
            
            # Find the question object
            question = next((q for q in sub_questions if q["id"] == question_id), None)
            if not question:
                return
                
            # Process dependencies first
            for dep_id in question.get("dependencies", []):
                process_question(dep_id)
                
            processed.add(question_id)
            sequence.append(question_id)
        
        # Process all questions
        for question in sub_questions:
            process_question(question["id"])
            
        return sequence
    
    def generate_reasoning_trace(self, query: str, sub_questions: List[Dict], answers: List[str]) -> str:
        """
        Generate a reasoning trace showing the Chain of Thought process.
        
        Args:
            query (str): The original complex query.
            sub_questions (List[Dict]): List of sub-questions.
            answers (List[str]): List of answers to sub-questions.
            
        Returns:
            str: The reasoning trace text.
        """
        trace = [f"Original Question: {query}\n\nChain of Thought Reasoning Process:"]
        
        for i, sq in enumerate(sub_questions):
            answer = answers[i] if i < len(answers) else "No answer available"
            trace.append(f"\nStep {i+1}: {sq['question']}")
            trace.append(f"Reasoning: {sq.get('relevance', 'No reasoning provided')}")
            trace.append(f"Answer: {answer}")
        
        return "\n".join(trace)  

class ReasoningTraceVisualizer:
    """
    Visualizer for reasoning traces and Chain of Thought processes.
    
    This class provides methods to convert reasoning processes into 
    structured visualizations for displaying in user interfaces.
    
    Attributes:
        trace_data (Dict[str, Any]): Data for the reasoning trace.
    """
    
    def __init__(self, trace_data: Optional[Dict[str, Any]] = None):
        """
        Initialize the reasoning trace visualizer.
        
        Args:
            trace_data (Optional[Dict[str, Any]]): Initial trace data to visualize.
        """
        self.trace_data = trace_data or {}
        self.visualization_formats = ["text", "html", "json", "markdown"]
    
    def set_trace_data(self, trace_data: Dict[str, Any]) -> None:
        """
        Set the trace data for visualization.
        
        Args:
            trace_data (Dict[str, Any]): Reasoning trace data to visualize.
        """
        self.trace_data = trace_data
    
    def format_trace_as_text(self) -> str:
        """
        Format the reasoning trace as plain text.
        
        Returns:
            str: Text representation of the reasoning trace.
        """
        if not self.trace_data:
            return "No reasoning trace data available."
        
        # Extract key components
        original_query = self.trace_data.get("original_query", "Unknown query")
        reasoning_steps = self.trace_data.get("reasoning_steps", [])
        sub_questions = self.trace_data.get("sub_questions", [])
        
        text_output = [
            "=== REASONING TRACE ===",
            f"Original Query: {original_query}",
            "\n=== REASONING STEPS ===",
        ]
        
        # Add reasoning steps
        for i, step in enumerate(reasoning_steps, 1):
            text_output.append(f"{i}. {step}")
        
        # Add sub-questions
        if sub_questions:
            text_output.append("\n=== SUB-QUESTIONS ===")
            for i, sq in enumerate(sub_questions, 1):
                question = sq.get("question", "Unknown")
                relevance = sq.get("relevance", "")
                deps = sq.get("dependencies", [])
                
                text_output.append(f"Sub-question {i}: {question}")
                if relevance:
                    text_output.append(f"   Relevance: {relevance}")
                if deps:
                    text_output.append(f"   Dependencies: {', '.join(map(str, deps))}")
                text_output.append("")
        
        return "\n".join(text_output)
    
    def format_trace_as_html(self) -> str:
        """
        Format the reasoning trace as HTML for web display.
        
        Returns:
            str: HTML representation of the reasoning trace.
        """
        if not self.trace_data:
            return "<p>No reasoning trace data available.</p>"
        
        # Extract key components
        original_query = self.trace_data.get("original_query", "Unknown query")
        reasoning_steps = self.trace_data.get("reasoning_steps", [])
        sub_questions = self.trace_data.get("sub_questions", [])
        
        html_output = [
            "<div class='reasoning-trace'>",
            f"<h3>Original Query</h3>",
            f"<p class='query'>{original_query}</p>",
            "<h3>Reasoning Steps</h3>",
            "<ol class='reasoning-steps'>",
        ]
        
        # Add reasoning steps
        for step in reasoning_steps:
            html_output.append(f"<li>{step}</li>")
        
        html_output.append("</ol>")
        
        # Add sub-questions
        if sub_questions:
            html_output.append("<h3>Sub-Questions</h3>")
            html_output.append("<div class='sub-questions'>")
            
            for i, sq in enumerate(sub_questions, 1):
                question = sq.get("question", "Unknown")
                relevance = sq.get("relevance", "")
                deps = sq.get("dependencies", [])
                
                html_output.append(f"<div class='sub-question' id='sq-{i}'>")
                html_output.append(f"<h4>Sub-question {i}</h4>")
                html_output.append(f"<p class='question'>{question}</p>")
                
                if relevance:
                    html_output.append(f"<p class='relevance'><strong>Relevance:</strong> {relevance}</p>")
                
                if deps:
                    dep_links = []
                    for dep in deps:
                        dep_links.append(f"<a href='#sq-{dep}'>Sub-question {dep}</a>")
                    
                    html_output.append(f"<p class='dependencies'><strong>Dependencies:</strong> {', '.join(dep_links)}</p>")
                
                html_output.append("</div>")
            
            html_output.append("</div>")
        
        html_output.append("</div>")
        
        return "\n".join(html_output)
    
    def format_trace_as_markdown(self) -> str:
        """
        Format the reasoning trace as Markdown.
        
        Returns:
            str: Markdown representation of the reasoning trace.
        """
        if not self.trace_data:
            return "No reasoning trace data available."
        
        # Extract key components
        original_query = self.trace_data.get("original_query", "Unknown query")
        reasoning_steps = self.trace_data.get("reasoning_steps", [])
        sub_questions = self.trace_data.get("sub_questions", [])
        
        md_output = [
            "# Reasoning Trace",
            f"## Original Query",
            f"> {original_query}",
            "",
            "## Reasoning Steps",
            "",
        ]
        
        # Add reasoning steps
        for i, step in enumerate(reasoning_steps, 1):
            md_output.append(f"{i}. {step}")
        
        # Add sub-questions
        if sub_questions:
            md_output.append("\n## Sub-Questions\n")
            for i, sq in enumerate(sub_questions, 1):
                question = sq.get("question", "Unknown")
                relevance = sq.get("relevance", "")
                deps = sq.get("dependencies", [])
                
                md_output.append(f"### Sub-question {i}: {question}")
                if relevance:
                    md_output.append(f"**Relevance**: {relevance}")
                if deps:
                    md_output.append(f"**Dependencies**: Sub-questions {', '.join(map(str, deps))}")
                md_output.append("")
        
        return "\n".join(md_output)
    
    def format_trace_as_json(self) -> str:
        """
        Format the reasoning trace as JSON.
        
        Returns:
            str: JSON representation of the reasoning trace.
        """
        return json.dumps(self.trace_data, indent=2)
    
    def generate_visualization(self, format_type: str = "text") -> str:
        """
        Generate a visualization of the reasoning trace in the specified format.
        
        Args:
            format_type (str): The format to use: 'text', 'html', 'json', or 'markdown'.
            
        Returns:
            str: The formatted reasoning trace.
            
        Raises:
            ValueError: If an invalid format is specified.
        """
        format_type = format_type.lower()
        
        if format_type not in self.visualization_formats:
            raise ValueError(f"Invalid format type. Supported formats: {', '.join(self.visualization_formats)}")
        
        if format_type == "text":
            return self.format_trace_as_text()
        elif format_type == "html":
            return self.format_trace_as_html()
        elif format_type == "markdown":
            return self.format_trace_as_markdown()
        elif format_type == "json":
            return self.format_trace_as_json()
        
        # Fallback to text if format not recognized (should not happen due to validation)
        return self.format_trace_as_text()
    
    def generate_interactive_visualization(self, trace_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate an interactive visualization data structure.
        
        This method creates a comprehensive data structure for building interactive
        visualizations of reasoning traces, including node and edge relationships for 
        graph-based visualizations.
        
        Args:
            trace_data (Optional[Dict[str, Any]]): Reasoning trace data to visualize. 
                                              If None, uses the current trace_data.
        
        Returns:
            Dict[str, Any]: Data structure for interactive visualization.
        """
        if trace_data:
            self.set_trace_data(trace_data)
        
        if not self.trace_data:
            return {"error": "No reasoning trace data available."}
        
        # Extract key components
        original_query = self.trace_data.get("original_query", "Unknown query")
        reasoning_steps = self.trace_data.get("reasoning_steps", [])
        sub_questions = self.trace_data.get("sub_questions", [])
        results = self.trace_data.get("results", {})
        
        # Create node data for graph visualization
        nodes = [
            {
                "id": "query",
                "type": "query",
                "label": "Original Query",
                "content": original_query
            }
        ]
        
        # Add reasoning steps as nodes
        for i, step in enumerate(reasoning_steps, 1):
            nodes.append({
                "id": f"step_{i}",
                "type": "step",
                "label": f"Step {i}",
                "content": step
            })
        
        # Add sub-questions as nodes
        for i, sq in enumerate(sub_questions, 1):
            sq_id = f"sq_{i}"
            nodes.append({
                "id": sq_id,
                "type": "sub_question",
                "label": f"Sub-question {i}",
                "content": sq.get("question", "Unknown"),
                "metadata": {
                    "relevance": sq.get("relevance", ""),
                    "dependencies": sq.get("dependencies", [])
                }
            })
            
            # Add result node if available
            if str(i) in results:
                result_id = f"result_{i}"
                nodes.append({
                    "id": result_id,
                    "type": "result",
                    "label": f"Result {i}",
                    "content": results[str(i)],
                    "parent": sq_id
                })
        
        # Create edges for dependencies
        edges = [
            {
                "source": "query",
                "target": "step_1",
                "type": "flow"
            }
        ]
        
        # Connect reasoning steps
        for i in range(1, len(reasoning_steps)):
            edges.append({
                "source": f"step_{i}",
                "target": f"step_{i+1}",
                "type": "flow"
            })
        
        # Connect last reasoning step to first sub-question
        if reasoning_steps and sub_questions:
            edges.append({
                "source": f"step_{len(reasoning_steps)}",
                "target": "sq_1",
                "type": "flow"
            })
        
        # Connect sub-questions based on dependencies
        for i, sq in enumerate(sub_questions, 1):
            sq_id = f"sq_{i}"
            
            # Connect to results
            if str(i) in results:
                edges.append({
                    "source": sq_id,
                    "target": f"result_{i}",
                    "type": "result"
                })
            
            # Connect dependencies
            for dep in sq.get("dependencies", []):
                if isinstance(dep, int) and 1 <= dep <= len(sub_questions):
                    edges.append({
                        "source": f"sq_{dep}",
                        "target": sq_id,
                        "type": "dependency"
                    })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "query": original_query,
                "step_count": len(reasoning_steps),
                "sub_question_count": len(sub_questions),
                "formats": self.visualization_formats
            }
        }  
