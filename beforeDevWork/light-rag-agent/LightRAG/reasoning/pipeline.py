from typing import Dict, List, Optional, Union, Any
import time
import json
import os
import uuid
from datetime import datetime
from .config import logger
from .agent import ReasoningAgent
from .models import DeepSeekModel
from .cot import ChainOfThought, ReasoningTraceVisualizer
from .optimizer import QueryOptimizer

class RetrievalMetadata:
    """
    Class for tracking metadata and performance metrics for retrieval operations.
    
    This class stores detailed metadata about each retrieval operation, including
    timing, parameters, and result statistics.
    
    Attributes:
        session_id (str): Unique identifier for the retrieval session.
        query (str): The original user query.
        sub_queries (List[Dict]): List of sub-queries with their metadata.
        start_time (float): Timestamp when the retrieval session started.
        end_time (float): Timestamp when the retrieval session completed.
        total_results (int): Total number of results retrieved.
        metadata_path (str): Path to store metadata files.
    """
    
    def __init__(self, query: str, metadata_path: Optional[str] = None):
        """
        Initialize the retrieval metadata tracker.
        
        Args:
            query (str): The original user query.
            metadata_path (Optional[str]): Path to store metadata files. If None, uses default.
        """
        self.session_id = str(uuid.uuid4())
        self.query = query
        self.sub_queries = []
        self.start_time = time.time()
        self.end_time = None
        self.total_results = 0
        
        # Set metadata storage path
        if metadata_path:
            self.metadata_path = metadata_path
        else:
            # Default path in the logs directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.metadata_path = os.path.join(os.path.dirname(current_dir), "logs", "metadata")
        
        # Ensure directory exists
        os.makedirs(self.metadata_path, exist_ok=True)
        
        # Log session start
        logger.info(f"Retrieval session {self.session_id} started for query: {query[:50]}...")
    
    def add_sub_query(self, sub_query_id: int, sub_query: str, parameters: Dict[str, Any]) -> None:
        """
        Add metadata for a sub-query retrieval operation.
        
        Args:
            sub_query_id (int): Identifier for the sub-query.
            sub_query (str): The sub-query text.
            parameters (Dict[str, Any]): Parameters used for retrieval.
        """
        self.sub_queries.append({
            "id": sub_query_id,
            "query": sub_query,
            "parameters": parameters,
            "start_time": time.time(),
            "end_time": None,
            "result_count": 0,
            "results": []
        })
    
    def record_results(self, sub_query_id: int, results: List[Dict[str, Any]]) -> None:
        """
        Record retrieval results for a sub-query.
        
        Args:
            sub_query_id (int): Identifier for the sub-query.
            results (List[Dict[str, Any]]): List of retrieval results.
        """
        # Find the sub-query entry
        for sq in self.sub_queries:
            if sq["id"] == sub_query_id:
                sq["end_time"] = time.time()
                sq["result_count"] = len(results)
                
                # Store result metadata but not full content to avoid excessive storage
                sq["results"] = [{
                    "id": i,
                    "source": result.get("source", "unknown"),
                    "score": result.get("score", 0),
                    "content_length": len(result.get("content", "")),
                } for i, result in enumerate(results)]
                
                # Update total results count
                self.total_results += len(results)
                break
    
    def complete_session(self) -> None:
        """
        Mark the retrieval session as complete and save metadata.
        """
        self.end_time = time.time()
        self.save_metadata()
        
        # Log session completion
        duration = self.end_time - self.start_time
        logger.info(f"Retrieval session {self.session_id} completed in {duration:.2f}s with {self.total_results} results")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the retrieval session.
        
        Returns:
            Dict[str, Any]: Summary of the retrieval session.
        """
        duration = (self.end_time or time.time()) - self.start_time
        return {
            "session_id": self.session_id,
            "query": self.query,
            "duration": duration,
            "sub_query_count": len(self.sub_queries),
            "total_results": self.total_results,
            "timestamp": datetime.now().isoformat()
        }
    
    def save_metadata(self) -> str:
        """
        Save the retrieval metadata to a file.
        
        Returns:
            str: Path to the saved metadata file.
        """
        # Create filename with timestamp and session ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"retrieval_metadata_{timestamp}_{self.session_id[:8]}.json"
        filepath = os.path.join(self.metadata_path, filename)
        
        # Prepare metadata object
        metadata = {
            "session_id": self.session_id,
            "query": self.query,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(self.end_time or time.time()).isoformat(),
            "duration": (self.end_time or time.time()) - self.start_time,
            "total_results": self.total_results,
            "sub_queries": self.sub_queries
        }
        
        # Write to file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        return filepath

class ResultsStorage:
    """
    Storage system for retrieval results and associated metadata.
    
    This class manages the storage, retrieval, and caching of query results,
    enabling efficient reuse of previously retrieved information.
    
    Attributes:
        cache_enabled (bool): Whether result caching is enabled.
        storage_path (str): Path to store result files.
        cache (Dict): In-memory cache of retrieval results.
        cache_hits (int): Number of cache hits.
        cache_misses (int): Number of cache misses.
    """
    
    def __init__(self, storage_path: Optional[str] = None, cache_enabled: bool = True, 
                 cache_size: int = 100):
        """
        Initialize the results storage system.
        
        Args:
            storage_path (Optional[str]): Path to store result files. If None, uses default.
            cache_enabled (bool): Whether to enable result caching.
            cache_size (int): Maximum number of queries to cache in memory.
        """
        self.cache_enabled = cache_enabled
        self.cache_size = cache_size
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Set storage path
        if storage_path:
            self.storage_path = storage_path
        else:
            # Default path in the logs directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.storage_path = os.path.join(os.path.dirname(current_dir), "logs", "results")
        
        # Ensure directory exists
        os.makedirs(self.storage_path, exist_ok=True)
        
        logger.info(f"Results storage initialized with cache {'enabled' if cache_enabled else 'disabled'}")
    
    def _get_cache_key(self, query: str, parameters: Dict[str, Any]) -> str:
        """
        Generate a cache key from query and parameters.
        
        Args:
            query (str): The query string.
            parameters (Dict[str, Any]): Retrieval parameters.
            
        Returns:
            str: A unique cache key.
        """
        # Simplified cache key based on query and sorted parameter keys
        params_str = json.dumps(parameters, sort_keys=True)
        key = f"{query}:{params_str}"
        return str(hash(key))
    
    def store_results(self, query: str, parameters: Dict[str, Any], 
                      results: List[Dict[str, Any]]) -> str:
        """
        Store retrieval results for a query.
        
        Args:
            query (str): The query string.
            parameters (Dict[str, Any]): Parameters used for retrieval.
            results (List[Dict[str, Any]]): List of retrieval results.
            
        Returns:
            str: Identifier for the stored results.
        """
        # Generate a unique ID for the results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
        
        # Create the result object
        result_obj = {
            "id": result_id,
            "query": query,
            "parameters": parameters,
            "timestamp": datetime.now().isoformat(),
            "results": results
        }
        
        # Save to file
        filename = f"results_{result_id}.json"
        filepath = os.path.join(self.storage_path, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result_obj, f, indent=2)
        
        # Store in cache if enabled
        if self.cache_enabled:
            cache_key = self._get_cache_key(query, parameters)
            
            # Manage cache size
            if len(self.cache) >= self.cache_size:
                # Remove oldest entry (simple approach)
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[cache_key] = {
                "id": result_id,
                "results": results,
                "timestamp": datetime.now().timestamp()
            }
        
        logger.debug(f"Stored results for query: {query[:30]}... with {len(results)} items")
        return result_id
    
    def get_results(self, query: str, parameters: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve results for a query, using cache if available.
        
        Args:
            query (str): The query string.
            parameters (Dict[str, Any]): Parameters used for retrieval.
            
        Returns:
            Optional[List[Dict[str, Any]]]: The retrieval results or None if not found.
        """
        if self.cache_enabled:
            cache_key = self._get_cache_key(query, parameters)
            
            if cache_key in self.cache:
                self.cache_hits += 1
                logger.debug(f"Cache hit for query: {query[:30]}...")
                return self.cache[cache_key]["results"]
        
        # Cache miss or cache disabled
        self.cache_misses += 1
        logger.debug(f"Cache miss for query: {query[:30]}...")
        return None
    
    def get_results_by_id(self, result_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve results by their ID.
        
        Args:
            result_id (str): The result identifier.
            
        Returns:
            Optional[Dict[str, Any]]: The complete result object or None if not found.
        """
        # Look for the result file
        filename = f"results_{result_id}.json"
        filepath = os.path.join(self.storage_path, filename)
        
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading results file {filepath}: {e}")
                return None
        else:
            logger.warning(f"Results file not found for ID: {result_id}")
            return None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the results cache.
        
        Returns:
            Dict[str, Any]: Cache statistics.
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = 0 if total_requests == 0 else (self.cache_hits / total_requests * 100)
        
        return {
            "cache_enabled": self.cache_enabled,
            "cache_size": len(self.cache),
            "max_cache_size": self.cache_size,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_percent": hit_rate,
            "total_requests": total_requests
        }
    
    def clear_cache(self) -> None:
        """Clear the in-memory results cache."""
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Results cache cleared")

class ReasoningPipeline:
    """
    Pipeline coordinating the reasoning agent and RAG system.
    
    This class manages the overall reasoning and retrieval process,
    coordinating the interaction between the reasoning agent, Chain of Thought
    decomposition, and RAG system.
    
    Attributes:
        reasoning_agent (ReasoningAgent): The reasoning agent for query analysis and processing.
        rag_system: The RAG system for document retrieval.
        query_optimizer (QueryOptimizer): Optimizer for query parameters.
        results_storage (ResultsStorage): Storage system for retrieval results.
        trace_visualizer (ReasoningTraceVisualizer): Visualizer for reasoning traces.
    """
    
    def __init__(self, reasoning_agent: Optional[ReasoningAgent] = None, rag_system: Any = None,
                 enable_caching: bool = True):
        """
        Initialize the reasoning pipeline.
        
        Args:
            reasoning_agent (Optional[ReasoningAgent]): The reasoning agent. If None, creates new instance.
            rag_system (Any): The RAG system for document retrieval.
            enable_caching (bool): Whether to enable result caching.
        """
        self.reasoning_agent = reasoning_agent or ReasoningAgent()
        self.rag_system = rag_system
        self.query_optimizer = QueryOptimizer()
        self.results_storage = ResultsStorage(cache_enabled=enable_caching)
        self.trace_visualizer = ReasoningTraceVisualizer()
        
        # If the RAG system is not provided, log a warning
        if not self.rag_system:
            logger.warning("RAG system not provided. Pipeline will only perform reasoning without retrieval.")
    
    def process(self, query: str) -> Dict[str, Any]:
        """
        Process user queries, execute reasoning and retrieval processes.
        
        This method manages the complete flow from query analysis through
        decomposition into sub-questions, retrieval for each sub-question,
        and integration of results.
        
        Args:
            query (str): The user's query.
            
        Returns:
            Dict[str, Any]: The complete reasoning and retrieval results.
        """
        # Start tracking metadata
        metadata_tracker = RetrievalMetadata(query)
        
        # Step 1: Analyze the query
        logger.info(f"Processing query: {query}")
        analysis = self.reasoning_agent.analyze_query(query)
        
        # Step 2: Determine if decomposition is needed
        if analysis.get("requires_decomposition", False):
            # Complex query requiring decomposition
            logger.info("Query requires decomposition")
            
            # Step 3: Decompose problem into sub-questions
            sub_questions = self.reasoning_agent.decompose_problem(query, analysis)
            logger.info(f"Decomposed into {len(sub_questions)} sub-questions")
            
            # Step 4: Optimize the number of sub-questions if needed
            if len(sub_questions) > self.query_optimizer.optimize_sub_question_count(query, analysis):
                # Too many sub-questions, keep only the most important ones
                logger.info("Optimizing number of sub-questions")
                # Simple approach: keep the first N questions
                max_questions = self.query_optimizer.optimize_sub_question_count(query, analysis)
                sub_questions = sub_questions[:max_questions]
                logger.info(f"Reduced to {len(sub_questions)} sub-questions")
            
            # Step 5: Determine the optimal processing sequence
            cot = ChainOfThought()
            sequence = cot.create_retrieval_sequence(sub_questions)
            logger.info(f"Processing sequence determined: {sequence}")
            
            # Step 6: Process each sub-question in sequence
            all_results = {}
            
            for sq_id in sequence:
                # Find the sub-question
                sq = next((q for q in sub_questions if q["id"] == sq_id), None)
                if not sq:
                    continue
                
                logger.info(f"Processing sub-question {sq_id}: {sq['question']}")
                
                # Process the sub-question
                sq_result = self.process_subproblem(sq, query, metadata_tracker)
                
                # Store the results
                all_results[str(sq_id)] = sq_result
            
            # Step 7: Integrate results
            final_answer = self.reasoning_agent.integrate_results(all_results, query, sub_questions)
            
            # Complete metadata tracking
            metadata_tracker.complete_session()
            
            # Step 8: Prepare and return results
            result = {
                "query": query,
                "analysis": analysis,
                "sub_questions": sub_questions,
                "sequence": sequence,
                "sub_question_results": all_results,
                "answer": final_answer,
                "metadata": metadata_tracker.get_summary()
            }
            
            # Create reasoning trace data for visualization
            trace_data = {
                "original_query": query,
                "reasoning_steps": [step["question"] for step in sub_questions],
                "sub_questions": sub_questions,
                "results": all_results
            }
            
            # Add visualization options
            result["visualization"] = {
                "text": self.trace_visualizer.generate_visualization(trace_data=trace_data, format_type="text"),
                "interactive_data": self.trace_visualizer.generate_interactive_visualization(trace_data)
            }
            
            return result
            
        else:
            # Simple query, no decomposition needed
            logger.info("Simple query, no decomposition needed")
            
            # Process as a single question
            sq = {"id": 1, "question": query}
            sq_result = self.process_subproblem(sq, query, metadata_tracker)
            
            # Complete metadata tracking
            metadata_tracker.complete_session()
            
            # Prepare and return results
            result = {
                "query": query,
                "analysis": analysis,
                "answer": sq_result,
                "metadata": metadata_tracker.get_summary()
            }
            
            return result
    
    def process_subproblem(self, subproblem: Dict[str, Any], original_query: str,
                          metadata_tracker: Optional[RetrievalMetadata] = None) -> Dict[str, Any]:
        """
        Process an individual sub-question and retrieve relevant content.
        
        Args:
            subproblem (Dict[str, Any]): The sub-question to process.
            original_query (str): The original user query.
            metadata_tracker (Optional[RetrievalMetadata]): Metadata tracker for recording results.
            
        Returns:
            Dict[str, Any]: The retrieval results for the sub-question.
        """
        sq_text = subproblem.get("question", "")
        sq_id = subproblem.get("id", 0)
        
        # Optimize retrieval parameters for this sub-question
        parameters = self.query_optimizer.optimize_retrieval_parameters(original_query, subproblem)
        
        # Record metadata if tracker provided
        if metadata_tracker:
            metadata_tracker.add_sub_query(sq_id, sq_text, parameters)
        
        # Check if we have cached results
        cached_results = self.results_storage.get_results(sq_text, parameters)
        if cached_results:
            logger.info(f"Using cached results for sub-question {sq_id}")
            
            # Record in metadata if tracker provided
            if metadata_tracker:
                metadata_tracker.record_results(sq_id, cached_results)
                
            return {
                "sub_question": sq_text,
                "parameters": parameters,
                "results": cached_results,
                "source": "cache"
            }
        
        # No cached results, perform retrieval
        if self.rag_system:
            try:
                # Call the RAG system with the sub-question and parameters
                logger.info(f"Retrieving content for sub-question {sq_id} with parameters: {parameters}")
                
                # Adapt this to match your RAG system's API
                retrieved_content = self.rag_system.retrieve(
                    query=sq_text,
                    **parameters
                )
                
                # Apply post-retrieval filtering
                filtered_results = self.query_optimizer.optimize_result_filtering(retrieved_content, sq_text)
                
                # Store results for future use
                self.results_storage.store_results(sq_text, parameters, filtered_results)
                
                # Record in metadata if tracker provided
                if metadata_tracker:
                    metadata_tracker.record_results(sq_id, filtered_results)
                
                return {
                    "sub_question": sq_text,
                    "parameters": parameters,
                    "results": filtered_results,
                    "source": "retrieval"
                }
                
            except Exception as e:
                logger.error(f"Error retrieving content for sub-question {sq_id}: {e}")
                
                # Return error information
                error_result = {
                    "sub_question": sq_text,
                    "parameters": parameters,
                    "error": str(e),
                    "results": [],
                    "source": "error"
                }
                
                # Record empty results in metadata if tracker provided
                if metadata_tracker:
                    metadata_tracker.record_results(sq_id, [])
                
                return error_result
        else:
            # No RAG system available
            logger.warning(f"No RAG system available for retrieval of sub-question {sq_id}")
            
            return {
                "sub_question": sq_text,
                "parameters": parameters,
                "results": [],
                "source": "no_rag_system"
            }
    
    def visualize_reasoning(self, reasoning_result: Dict[str, Any], format_type: str = "text") -> str:
        """
        Visualize the reasoning process and chain of thought.
        
        Args:
            reasoning_result (Dict[str, Any]): The reasoning process results.
            format_type (str): The format for visualization ('text', 'html', 'json', 'markdown')
            
        Returns:
            str: The visualization of the reasoning process.
        """
        # Extract relevant components for visualization
        query = reasoning_result.get("query", "Unknown query")
        sub_questions = reasoning_result.get("sub_questions", [])
        
        if "reasoning_steps" in reasoning_result:
            reasoning_steps = reasoning_result["reasoning_steps"]
        else:
            # If reasoning_steps not explicitly provided, use sub-questions
            reasoning_steps = [sq.get("question", "Unknown") for sq in sub_questions]
        
        # Create trace data
        trace_data = {
            "original_query": query,
            "reasoning_steps": reasoning_steps,
            "sub_questions": sub_questions,
            "results": reasoning_result.get("sub_question_results", {})
        }
        
        # Generate visualization
        return self.trace_visualizer.generate_visualization(trace_data=trace_data, format_type=format_type)
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the results storage and cache.
        
        Returns:
            Dict[str, Any]: Storage and cache statistics.
        """
        cache_stats = self.results_storage.get_cache_stats()
        
        return {
            "cache_stats": cache_stats,
            "optimization_cache": self.query_optimizer.get_cache_stats()
        }
    
    def clear_caches(self) -> None:
        """Clear all caches in the pipeline."""
        self.results_storage.clear_cache()
        self.query_optimizer.clear_cache()
        logger.info("All pipeline caches cleared") 
