from typing import Dict, List, Optional, Union, Any, Tuple
import json
import time
import logging
from .models import DeepSeekModel
from .cot import ChainOfThought
from .config import logger, ReasoningError

class ReasoningStepLogger:
    """
    Logger for tracking and visualizing reasoning steps.
    
    This class handles detailed logging of each step in the reasoning process,
    making it possible to debug, analyze, and visualize the entire
    reasoning chain.
    
    Attributes:
        logger (logging.Logger): Logger instance for recording steps.
        steps (List[Dict]): List of recorded reasoning steps.
        current_session_id (str): Identifier for the current reasoning session.
        start_time (float): Start time of the current reasoning session.
    """
    
    def __init__(self, log_level: int = logging.DEBUG):
        """
        Initialize the reasoning step logger.
        
        Args:
            log_level (int): Logging level to use for step logging.
        """
        self.logger = logger
        self.steps = []
        self.current_session_id = None
        self.start_time = None
        self.detailed_logs_enabled = True
    
    def start_session(self, query: str, session_id: Optional[str] = None) -> str:
        """
        Start a new reasoning session.
        
        Args:
            query (str): The user query that started this reasoning session.
            session_id (Optional[str]): Custom session ID. If None, uses timestamp.
            
        Returns:
            str: The session ID.
        """
        self.current_session_id = session_id or f"session_{int(time.time())}"
        self.start_time = time.time()
        self.steps = []
        
        # Record session start
        self.logger.info(f"Reasoning session {self.current_session_id} started for query: {query[:50]}...")
        
        # Record initial step
        self.log_step("session_start", {
            "query": query,
            "timestamp": time.time(),
            "session_id": self.current_session_id
        })
        
        return self.current_session_id
    
    def log_step(self, step_name: str, data: Dict[str, Any]) -> None:
        """
        Log a reasoning step with detailed data.
        
        Args:
            step_name (str): Name of the reasoning step.
            data (Dict[str, Any]): Data associated with this step.
        """
        # Ensure we have a current session
        if not self.current_session_id:
            self.start_session("Unknown query")
        
        # Create step record
        step = {
            "step_name": step_name,
            "timestamp": time.time(),
            "elapsed": time.time() - self.start_time,
            "data": data
        }
        
        # Add to steps list
        self.steps.append(step)
        
        # Log basic info
        self.logger.debug(f"Reasoning step: {step_name} | Elapsed: {step['elapsed']:.2f}s")
        
        # Optionally log detailed data
        if self.detailed_logs_enabled:
            try:
                # Format the data as JSON for structured logging
                if isinstance(data, dict):
                    # For large data structures, limit the output
                    compact_data = {}
                    for k, v in data.items():
                        if isinstance(v, str) and len(v) > 200:
                            compact_data[k] = v[:200] + "..."
                        elif isinstance(v, list) and len(v) > 5:
                            compact_data[k] = v[:5] + ["..."]
                        else:
                            compact_data[k] = v
                            
                    self.logger.debug(f"Step data: {json.dumps(compact_data, indent=2)}")
                else:
                    self.logger.debug(f"Step data: {data}")
            except Exception as e:
                self.logger.warning(f"Failed to log detailed step data: {e}")
    
    def end_session(self) -> List[Dict[str, Any]]:
        """
        End the current reasoning session and return all recorded steps.
        
        Returns:
            List[Dict[str, Any]]: All recorded reasoning steps.
        """
        if not self.current_session_id:
            return []
            
        # Calculate total duration
        total_duration = time.time() - self.start_time
        
        # Log session end
        self.logger.info(f"Reasoning session {self.current_session_id} completed in {total_duration:.2f}s with {len(self.steps)} steps")
        
        # Record final step
        self.log_step("session_end", {
            "total_steps": len(self.steps),
            "total_duration": total_duration
        })
        
        # Return all steps
        return self.steps
    
    def get_step_timeline(self) -> List[Dict[str, Any]]:
        """
        Get a timeline of all reasoning steps with timing information.
        
        Returns:
            List[Dict[str, Any]]: Timeline of reasoning steps.
        """
        timeline = []
        
        for i, step in enumerate(self.steps):
            timeline.append({
                "step_number": i + 1,
                "step_name": step["step_name"],
                "elapsed_time": step["elapsed"],
                "timestamp": step["timestamp"]
            })
            
        return timeline
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current reasoning session.
        
        Returns:
            Dict[str, Any]: Summary of the reasoning session.
        """
        if not self.steps:
            return {"status": "No steps recorded"}
            
        return {
            "session_id": self.current_session_id,
            "started_at": self.start_time,
            "total_steps": len(self.steps),
            "total_duration": time.time() - self.start_time,
            "steps": [step["step_name"] for step in self.steps]
        }

class ReasoningAgent:
    """
    Core reasoning agent for complex query analysis and decomposition.
    
    This class handles the main reasoning functionality, including
    query analysis, problem decomposition, and integration of
    retrieval results.
    
    Attributes:
        model (DeepSeekModel): Language model for reasoning operations.
        cot (ChainOfThought): Chain of Thought reasoning module.
        step_logger (ReasoningStepLogger): Logger for reasoning steps.
    """
    
    def __init__(self, model: Optional[DeepSeekModel] = None):
        """
        Initialize the reasoning agent.
        
        Args:
            model (Optional[DeepSeekModel]): Language model to use. If None, creates a new instance.
        """
        self.model = model or DeepSeekModel()
        self.cot = ChainOfThought(model=self.model)
        self.current_subproblems = []
        self.step_logger = ReasoningStepLogger()
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze the complexity, type, and key concepts of the query.
        
        Args:
            query (str): The user query to analyze.
            
        Returns:
            Dict[str, Any]: Analysis results including complexity level, 
                           question type, key concepts, etc.
        """
        # Start logging for this query
        self.step_logger.start_session(query)
        self.step_logger.log_step("analyze_query_start", {"query": query})
        
        prompt = f"""
        Please analyze this question: "{query}"
        
        Provide an analysis with the following information:
        1. Complexity level (simple, moderate, complex)
        2. Question type (factoid, comparative, exploratory, causal, etc.)
        3. Key concepts and entities mentioned
        4. Whether the question requires breaking down into sub-questions
        5. Domain or knowledge areas relevant to the question
        
        Format your response as a structured analysis that can be easily parsed.
        """
        
        try:
            start_time = time.time()
            response = self.model.call(prompt, {"temperature": 0.2})
            
            # Log the raw response
            self.step_logger.log_step("analyze_query_raw_response", {"response": response})
            
            # Parse the analysis from the response
            analysis = self._parse_analysis(response)
            
            # Extract numerical complexity for easier processing
            complexity = analysis.get("complexity", "").lower()
            if "simple" in complexity:
                analysis["complexity_score"] = 1
            elif "moderate" in complexity:
                analysis["complexity_score"] = 2
            elif "complex" in complexity:
                analysis["complexity_score"] = 3
            else:
                analysis["complexity_score"] = 2  # Default to moderate
                
            # Add performance metrics
            analysis["processing_time"] = time.time() - start_time
            
            # Log completion
            self.step_logger.log_step("analyze_query_complete", {"analysis": analysis})
            
            return analysis
            
        except Exception as e:
            self.step_logger.log_step("analyze_query_error", {"error": str(e)})
            logger.error(f"Error analyzing query: {e}")
            raise ReasoningError(f"Failed to analyze query: {str(e)}", step="analyze_query", data={"query": query})
    
    def _parse_analysis(self, response: str) -> Dict[str, Any]:
        """
        Parse the analysis from the model's response.
        
        Args:
            response (str): The model's response to the analysis prompt.
            
        Returns:
            Dict[str, Any]: Structured analysis data.
        """
        analysis = {
            "complexity": "moderate",  # Default values
            "question_type": "unknown",
            "key_concepts": [],
            "requires_decomposition": False,
            "domains": []
        }
        
        try:
            # Check for complexity level
            if "complexity level" in response.lower():
                for line in response.split("\n"):
                    if "complexity" in line.lower():
                        if "simple" in line.lower():
                            analysis["complexity"] = "simple"
                        elif "moderate" in line.lower():
                            analysis["complexity"] = "moderate"
                        elif "complex" in line.lower():
                            analysis["complexity"] = "complex"
            
            # Check for question type
            if "question type" in response.lower():
                for line in response.split("\n"):
                    if "question type" in line.lower() or "type:" in line.lower():
                        parts = line.split(":", 1)
                        if len(parts) > 1:
                            q_type = parts[1].strip().lower()
                            if any(t in q_type for t in ["factoid", "comparative", "exploratory", "causal", "procedural", "hypothetical"]):
                                analysis["question_type"] = q_type
            
            # Extract key concepts
            if "key concepts" in response.lower():
                for line in response.split("\n"):
                    if "key concepts" in line.lower() or "entities" in line.lower():
                        parts = line.split(":", 1)
                        if len(parts) > 1:
                            concepts = parts[1].strip()
                            # Split on common separators
                            if "," in concepts:
                                analysis["key_concepts"] = [c.strip() for c in concepts.split(",")]
                            else:
                                analysis["key_concepts"] = [c.strip() for c in concepts.split()]
            
            # Check if decomposition is needed
            if "sub-question" in response.lower():
                for line in response.split("\n"):
                    if "requires breaking down" in line.lower() or "sub-question" in line.lower():
                        analysis["requires_decomposition"] = "yes" in line.lower() or "true" in line.lower() or "require" in line.lower()
            
            # Extract domains
            if "domain" in response.lower():
                for line in response.split("\n"):
                    if "domain" in line.lower() or "knowledge area" in line.lower():
                        parts = line.split(":", 1)
                        if len(parts) > 1:
                            domains = parts[1].strip()
                            if "," in domains:
                                analysis["domains"] = [d.strip() for d in domains.split(",")]
                            else:
                                analysis["domains"] = [d.strip() for d in domains.split()]
                                
            # Add the full response for reference
            analysis["full_response"] = response
            
            # Log the parsed analysis
            self.step_logger.log_step("parse_analysis", {"parsed": analysis})
            
        except Exception as e:
            self.step_logger.log_step("parse_analysis_error", {"error": str(e), "response": response})
            logger.warning(f"Error parsing analysis: {e}")
            # Return the default analysis on error
        
        return analysis
        
    def decompose_problem(self, query: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Break down the problem into multiple sub-questions using Chain of Thought reasoning.
        
        Args:
            query (str): The original user query.
            analysis (Dict[str, Any]): Analysis of the query from analyze_query.
            
        Returns:
            List[Dict[str, Any]]: List of sub-questions with metadata.
        """
        self.step_logger.log_step("decompose_problem_start", {
            "query": query,
            "analysis": analysis
        })
        
        # Determine if decomposition is needed based on analysis
        requires_decomposition = analysis.get("requires_decomposition", False)
        
        # For simple queries, we might return a single question
        if not requires_decomposition and analysis.get("complexity", "") == "simple":
            self.step_logger.log_step("no_decomposition_needed", {
                "reason": "Simple query, no decomposition needed",
                "complexity": analysis.get("complexity", "")
            })
            
            # Return the original query as the only sub-question
            single_question = [{
                "id": 1,
                "question": query,
                "relevance": "This is the original question.",
                "dependencies": [],
                "is_original": True
            }]
            
            self.current_subproblems = single_question
            
            self.step_logger.log_step("decompose_problem_complete", {
                "sub_questions": single_question,
                "count": 1
            })
            
            return single_question
        
        try:
            # Use the Chain of Thought module to decompose the question
            start_time = time.time()
            
            self.step_logger.log_step("cot_decomposition_start", {})
            
            # Determine the maximum number of sub-questions based on complexity
            max_sub_questions = 3  # Default
            if analysis.get("complexity") == "complex":
                max_sub_questions = 5
            elif analysis.get("complexity") == "moderate":
                max_sub_questions = 3
            
            sub_questions = self.cot.decompose_question(query)
            
            self.step_logger.log_step("cot_decomposition_complete", {
                "count": len(sub_questions),
                "processing_time": time.time() - start_time
            })
            
            # Save the sub-questions for later use
            self.current_subproblems = sub_questions
            
            # Log each sub-question
            for i, sq in enumerate(sub_questions):
                self.step_logger.log_step(f"sub_question_{i+1}", {
                    "question": sq.get("question", ""),
                    "relevance": sq.get("relevance", ""),
                    "dependencies": sq.get("dependencies", [])
                })
            
            self.step_logger.log_step("decompose_problem_complete", {
                "sub_questions_count": len(sub_questions)
            })
            
            return sub_questions
            
        except Exception as e:
            self.step_logger.log_step("decompose_problem_error", {"error": str(e)})
            logger.error(f"Error decomposing problem: {e}")
            raise ReasoningError(
                f"Failed to decompose problem: {str(e)}",
                step="decompose_problem",
                data={"query": query, "analysis": analysis}
            )
    
    def determine_strategy(self, subproblem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine the best retrieval strategy for a specific sub-question.
        
        Args:
            subproblem (Dict[str, Any]): The sub-question to determine strategy for.
            
        Returns:
            Dict[str, Any]: Retrieval strategy parameters.
        """
        self.step_logger.log_step("determine_strategy_start", {
            "sub_question": subproblem.get("question", ""),
            "sub_question_id": subproblem.get("id", 0)
        })
        
        question = subproblem.get("question", "")
        
        # Define a default strategy
        default_strategy = {
            "top_k": 5,
            "method": "vector_search",
            "filters": {}
        }
        
        # Adjust strategy based on question characteristics
        # These are just simple heuristics - in a real system, we would use
        # more sophisticated analysis
        
        # For entity-focused questions, use more targeted retrieval
        if any(entity_word in question.lower() for entity_word in ["who", "when", "where", "name"]):
            default_strategy["top_k"] = 3
            
        # For broader questions, cast a wider net
        if any(broad_word in question.lower() for broad_word in ["why", "how", "explain", "describe"]):
            default_strategy["top_k"] = 8
            
        # For relationship queries, consider knowledge graph if available
        if any(rel_word in question.lower() for rel_word in ["related", "connection", "relationship", "between"]):
            default_strategy["method"] = "knowledge_graph"
            
        # Log and return the strategy
        self.step_logger.log_step("determine_strategy_complete", {
            "strategy": default_strategy
        })
        
        return default_strategy
    
    def integrate_results(self, subproblem_results: Dict[str, Any], original_query: str, 
                         sub_questions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Integrate results from multiple sub-question retrievals into a coherent answer.
        
        Args:
            subproblem_results (Dict[str, Any]): Results for each sub-question.
            original_query (str): The original user query.
            sub_questions (Optional[List[Dict[str, Any]]]): List of sub-questions. If None, uses current_subproblems.
            
        Returns:
            Dict[str, Any]: Integrated results including the final answer.
        """
        self.step_logger.log_step("integrate_results_start", {
            "original_query": original_query,
            "num_results": len(subproblem_results)
        })
        
        # Use provided sub_questions or fall back to current_subproblems
        sub_questions = sub_questions or self.current_subproblems
        
        # Extract retrieved information for each sub-question
        retrieved_info = []
        
        for sq_id, result in subproblem_results.items():
            # Find the corresponding sub-question
            sq = next((q for q in sub_questions if str(q.get("id", "")) == sq_id), None)
            
            if not sq:
                continue
                
            # Get results from this sub-question
            sq_results = result.get("results", [])
            
            # Add to the retrieved information
            retrieved_info.append(f"Sub-question: {sq.get('question', '')}")
            
            if not sq_results:
                retrieved_info.append("No relevant information found for this sub-question.")
            else:
                for i, item in enumerate(sq_results[:3]):  # Only include first 3 results
                    content = item.get("content", "")
                    source = item.get("source", "Unknown source")
                    retrieved_info.append(f"Information {i+1} (from {source}): {content[:200]}...")
            
            retrieved_info.append("")  # Add blank line between sub-questions
            
        # Log the retrieved information
        self.step_logger.log_step("retrieved_information", {
            "count": len(retrieved_info),
            "info": retrieved_info[:5] + (["..."] if len(retrieved_info) > 5 else [])
        })
        
        # Prepare the integration prompt
        integration_prompt = f"""
        Original question: {original_query}
        
        I have retrieved information for several sub-questions to help answer this question.
        Please integrate this information into a coherent, comprehensive answer.
        
        Retrieved information:
        {chr(10).join(retrieved_info)}
        
        Please provide:
        1. A direct answer to the original question
        2. Supporting details from the retrieved information
        3. Any remaining uncertainties or aspects not covered by the retrieved information
        """
        
        try:
            # Generate the integrated answer
            start_time = time.time()
            integrated_answer = self.model.call(integration_prompt, {"temperature": 0.3})
            processing_time = time.time() - start_time
            
            # Log the integration result
            self.step_logger.log_step("integration_complete", {
                "processing_time": processing_time,
                "answer_length": len(integrated_answer)
            })
            
            # Complete the reasoning session
            self.step_logger.end_session()
            
            return {
                "answer": integrated_answer,
                "processing_time": processing_time,
                "retrieved_info_count": len(retrieved_info),
                "sub_questions_count": len(sub_questions),
                "reasoning_timeline": self.step_logger.get_step_timeline()
            }
            
        except Exception as e:
            self.step_logger.log_step("integration_error", {"error": str(e)})
            logger.error(f"Error integrating results: {e}")
            
            # End the session even on error
            self.step_logger.end_session()
            
            raise ReasoningError(
                f"Failed to integrate results: {str(e)}",
                step="integrate_results",
                data={"query": original_query}
            )
    
    def execute_reasoning(self, query: str) -> Dict[str, Any]:
        """
        Execute the complete reasoning process, returning reasoning results and retrieval strategies.
        
        Args:
            query (str): The user query to process.
            
        Returns:
            Dict[str, Any]: Complete reasoning results including analysis, sub-questions, and strategies.
        """
        self.step_logger.start_session(query)
        self.step_logger.log_step("execute_reasoning_start", {"query": query})
        
        try:
            # Step 1: Analyze the query
            analysis = self.analyze_query(query)
            
            # Step 2: Decompose into sub-questions
            sub_questions = self.decompose_problem(query, analysis)
            
            # Step 3: Determine retrieval strategies for each sub-question
            strategies = {}
            for sq in sub_questions:
                sq_id = sq.get("id", 0)
                strategies[str(sq_id)] = self.determine_strategy(sq)
            
            # Log completion
            self.step_logger.log_step("execute_reasoning_complete", {
                "sub_questions_count": len(sub_questions),
                "strategies_count": len(strategies)
            })
            
            # Complete the reasoning session
            reasoning_timeline = self.step_logger.end_session()
            
            # Return the complete reasoning results
            return {
                "query": query,
                "analysis": analysis,
                "sub_questions": sub_questions,
                "strategies": strategies,
                "reasoning_timeline": reasoning_timeline
            }
            
        except Exception as e:
            self.step_logger.log_step("execute_reasoning_error", {"error": str(e)})
            logger.error(f"Error executing reasoning: {e}")
            
            # End the session even on error
            self.step_logger.end_session()
            
            raise ReasoningError(
                f"Failed to execute reasoning: {str(e)}",
                step="execute_reasoning",
                data={"query": query}
            )
            
    def get_reasoning_timeline(self) -> List[Dict[str, Any]]:
        """
        Get the timeline of reasoning steps from the current session.
        
        Returns:
            List[Dict[str, Any]]: Reasoning steps timeline.
        """
        return self.step_logger.get_step_timeline()
        
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current reasoning session.
        
        Returns:
            Dict[str, Any]: Session summary.
        """
        return self.step_logger.get_session_summary()  
