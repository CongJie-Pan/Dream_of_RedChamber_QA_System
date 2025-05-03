"""
Optimization module for the BasicRAG reasoning agent.

This module provides optimization utilities for the reasoning agent,
including sub-question count optimization and context-aware parameter
adjustment for retrieval operations.
"""

import json
import hashlib
import re
from typing import Dict, List, Optional, Any, Tuple, Set
from .config import logger, load_custom_prompts, MAX_SUB_QUESTIONS
from .models import DeepSeekModel

class QueryOptimizer:
    """
    Optimizer for query processing parameters.
    
    This class provides methods to optimize various aspects of the reasoning process,
    including the number of sub-questions and retrieval parameters based on context.
    
    Attributes:
        model (DeepSeekModel): The language model for optimization tasks.
        prompts (Dict[str, str]): Custom prompts for optimization tasks.
        optimization_cache (Dict): Cache for optimization results to avoid repeat calls.
    """
    
    def __init__(self, model: Optional[DeepSeekModel] = None):
        """
        Initialize the query optimizer.
        
        Args:
            model (Optional[DeepSeekModel]): The language model for optimization. If None, creates a new instance.
        """
        self.model = model or DeepSeekModel()
        self.prompts = load_custom_prompts()
        self.optimization_cache = {
            "sub_question_count": {},
            "retrieval_params": {},
            "filtering": {}
        }
    
    def _get_cache_key(self, query: str, extra_data: Optional[Any] = None) -> str:
        """
        Generate a cache key for optimization results.
        
        Args:
            query (str): The query string.
            extra_data (Optional[Any]): Additional data to include in key generation.
            
        Returns:
            str: A unique cache key.
        """
        key_data = query.lower().strip()
        if extra_data:
            if isinstance(extra_data, dict):
                # Sort dictionary keys for consistent hashing
                key_data += json.dumps(extra_data, sort_keys=True)
            else:
                key_data += str(extra_data)
                
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def optimize_sub_question_count(self, query: str, analysis: Dict[str, Any]) -> int:
        """
        Determine the optimal number of sub-questions for a given query.
        
        Uses a combination of heuristics and model-based optimization to determine
        the ideal number of sub-questions for breaking down a complex query.
        
        Args:
            query (str): The original query.
            analysis (Dict[str, Any]): Analysis of the query from the reasoning agent.
            
        Returns:
            int: The recommended number of sub-questions.
        """
        # Check cache first
        cache_key = self._get_cache_key(query, analysis)
        if cache_key in self.optimization_cache["sub_question_count"]:
            logger.debug(f"Using cached sub-question count for query: {query[:30]}...")
            return self.optimization_cache["sub_question_count"][cache_key]
        
        # Default value if optimization fails
        default_count = min(3, MAX_SUB_QUESTIONS)
        
        # Simple heuristic based on complexity
        complexity_based_count = None
        if analysis.get("complexity") == "simple":
            complexity_based_count = 1
        elif analysis.get("complexity") == "moderate":
            complexity_based_count = min(3, MAX_SUB_QUESTIONS)
        elif analysis.get("complexity") == "complex":
            complexity_based_count = min(5, MAX_SUB_QUESTIONS)
        
        # Count entities and key concepts if available
        entity_based_count = None
        if "key_concepts" in analysis and isinstance(analysis["key_concepts"], list):
            concepts_count = len(analysis["key_concepts"])
            if concepts_count > 0:
                # Use number of key concepts as a guide, with some normalization
                entity_based_count = max(1, min(MAX_SUB_QUESTIONS, 1 + (concepts_count // 2)))
        
        # Advanced analysis: look for multiple question markers in the query
        question_marker_count = None
        question_phrases = re.findall(r'(what|when|where|who|why|how|which|is|are|can|could|would|should|do|does)', 
                                    query.lower())
        if question_phrases:
            # Count unique question phrases, but normalize the result
            unique_phrases = set(question_phrases)
            if len(unique_phrases) > 1:
                question_marker_count = min(MAX_SUB_QUESTIONS, len(unique_phrases))
            
        # If optimization is enabled and we have the prompt, use the model
        model_based_count = None
        prompt_template = self.prompts.get("sub_question_count_optimization")
        if prompt_template:
            try:
                # Format the prompt with the query and analysis
                analysis_str = json.dumps(analysis)
                prompt = prompt_template.format(query=query, analysis=analysis_str)
                
                # Get the model's recommendation
                response = self.model.call(prompt, {"temperature": 0.1})
                
                # Try to extract a number from the response
                for line in response.split('\n'):
                    line = line.strip()
                    words = line.split()
                    for word in words:
                        if word.isdigit():
                            count = int(word)
                            if 1 <= count <= MAX_SUB_QUESTIONS:
                                model_based_count = count
                                break
                    if model_based_count:
                        break
                
                if not model_based_count:
                    logger.warning(f"Could not extract sub-question count from response: {response}")
            except Exception as e:
                logger.error(f"Error optimizing sub-question count: {e}")
        
        # Determine final count using all available methods
        # Prioritize: model > complexity > entities > question markers > default
        optimal_count = model_based_count or complexity_based_count or entity_based_count or question_marker_count or default_count
        
        # Log the decision process
        logger.debug(f"Sub-question count decision: model={model_based_count}, complexity={complexity_based_count}, "
                     f"entities={entity_based_count}, question_markers={question_marker_count}, final={optimal_count}")
        
        # Cache the result
        self.optimization_cache["sub_question_count"][cache_key] = optimal_count
        
        return optimal_count
    
    def optimize_retrieval_parameters(self, query: str, sub_question: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize retrieval parameters based on query and sub-question context.
        
        Args:
            query (str): The original query.
            sub_question (Dict[str, Any]): The sub-question to optimize parameters for.
            
        Returns:
            Dict[str, Any]: Optimized retrieval parameters.
        """
        sq_text = sub_question.get("question", "")
        
        # Check cache first
        cache_key = self._get_cache_key(query, sq_text)
        if cache_key in self.optimization_cache["retrieval_params"]:
            logger.debug(f"Using cached retrieval parameters for sub-question: {sq_text[:30]}...")
            return self.optimization_cache["retrieval_params"][cache_key]
            
        # Default parameters for BasicRAG (vector-only)
        default_params = {
            "retrieval_method": "vector",
            "top_k": 5,
            "similarity_threshold": 0.6,
            "depth": "medium",
            "filters": {}
        }
        
        # For questions about specific entities, use more targeted retrieval
        if "who" in sq_text.lower() or "when" in sq_text.lower() or "where" in sq_text.lower():
            default_params["top_k"] = 3
            default_params["similarity_threshold"] = 0.7
            
        # For broader questions, cast a wider net
        if "why" in sq_text.lower() or "how" in sq_text.lower() or "explain" in sq_text.lower():
            default_params["top_k"] = 8
            default_params["similarity_threshold"] = 0.5
            
        # If this is a follow-up/dependent question, adjust depth
        if sub_question.get("dependencies", []):
            default_params["depth"] = "deep"
            
        # If the sub-question is asking for examples or similar items, adjust parameters
        if "example" in sq_text.lower() or "similar" in sq_text.lower() or "like" in sq_text.lower():
            default_params["top_k"] = max(default_params["top_k"], 6)
            default_params["similarity_threshold"] = min(default_params["similarity_threshold"], 0.55)
            
        # If the sub-question mentions specific time periods or dates, add temporal filters
        date_pattern = re.compile(r'\b(in|during|before|after|between)\s+(\d{4}|\d{1,2}(st|nd|rd|th)\s+century|ancient|modern)\b', 
                                 re.IGNORECASE)
        if date_pattern.search(sq_text):
            default_params["filters"]["temporal"] = True
            
        # Use the model for parameter optimization if the prompt is available
        prompt_template = self.prompts.get("parameter_optimization")
        if prompt_template:
            try:
                prompt = prompt_template.format(
                    query=query, 
                    sub_question=sq_text
                )
                
                response = self.model.call(prompt, {"temperature": 0.1})
                
                # Try to parse the response as JSON
                try:
                    # Find JSON-like content in the response
                    start_idx = response.find('{')
                    end_idx = response.rfind('}') + 1
                    
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = response[start_idx:end_idx]
                        params = json.loads(json_str)
                        
                        # Validate and merge with defaults
                        if isinstance(params, dict):
                            for key, value in params.items():
                                # For BasicRAG, ensure retrieval_method stays as vector
                                if key == "retrieval_method":
                                    continue
                                if key in default_params:
                                    default_params[key] = value
                                    
                            logger.debug(f"Optimized parameters for sub-question: {default_params}")
                except Exception as json_e:
                    logger.warning(f"Failed to parse parameter JSON: {json_e}")
            except Exception as e:
                logger.error(f"Error in parameter optimization: {e}")
                
        # Cache the result
        self.optimization_cache["retrieval_params"][cache_key] = default_params
        
        return default_params
    
    def optimize_result_filtering(self, results: List[Dict[str, Any]], query: str = "") -> List[Dict[str, Any]]:
        """
        Filter and deduplicate retrieval results.
        
        Applies intelligent filtering, deduplication, and relevance sorting to 
        improve the quality of retrieval results.
        
        Args:
            results (List[Dict[str, Any]]): Raw retrieval results.
            query (str): The original query for relevance sorting.
            
        Returns:
            List[Dict[str, Any]]: Filtered and deduplicated results.
        """
        if not results:
            return []
            
        # Track metrics for logging
        original_count = len(results)
        
        # Enhanced deduplication using content similarity
        filtered_results = []
        seen_contents = set()
        
        # First pass: exact duplicate removal
        for result in results:
            content = result.get("content", "")
            
            # Create a simplified representation for similarity checking
            # Here we use the first 100 characters, but more sophisticated
            # approaches could be used (e.g., embedding similarity)
            content_key = content[:100].lower()
            
            if content_key not in seen_contents:
                seen_contents.add(content_key)
                filtered_results.append(result)
        
        # Second pass: near-duplicate detection using chunks
        if len(filtered_results) > 1:
            chunk_similarity_threshold = 0.8  # Similarity threshold for chunks
            chunk_size = 150
            
            chunked_results = []
            seen_chunks: Set[str] = set()
            
            for result in filtered_results:
                content = result.get("content", "")
                content_lower = content.lower()
                
                # Split content into chunks with overlap
                chunks = []
                for i in range(0, len(content_lower), chunk_size // 2):
                    chunk = content_lower[i:i + chunk_size]
                    if len(chunk) >= chunk_size // 2:  # Only consider substantial chunks
                        chunks.append(chunk)
                
                # Check if any chunk has been seen before
                is_near_duplicate = False
                new_chunks = []
                
                for chunk in chunks:
                    # For each chunk, check if it's similar to any seen chunk
                    chunk_hash = hashlib.md5(chunk.encode()).hexdigest()
                    
                    if chunk_hash in seen_chunks:
                        is_near_duplicate = True
                        break
                    new_chunks.append(chunk_hash)
                
                # Only add if it's not a near duplicate
                if not is_near_duplicate:
                    chunked_results.append(result)
                    seen_chunks.update(new_chunks)
            
            filtered_results = chunked_results
        
        # Third pass: prioritize results by relevance score if available
        if filtered_results and all("score" in result for result in filtered_results):
            filtered_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Log filtering metrics
        logger.debug(f"Filtered {original_count - len(filtered_results)} of {original_count} results "
                     f"({((original_count - len(filtered_results)) / original_count * 100):.1f}%)")
        
        return filtered_results
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the optimization cache.
        
        Returns:
            Dict[str, Any]: Cache statistics.
        """
        return {
            "sub_question_count_cache_size": len(self.optimization_cache["sub_question_count"]),
            "retrieval_params_cache_size": len(self.optimization_cache["retrieval_params"]),
            "filtering_cache_size": len(self.optimization_cache["filtering"]),
        }
        
    def clear_cache(self) -> None:
        """Clear all cached optimization results."""
        self.optimization_cache = {
            "sub_question_count": {},
            "retrieval_params": {},
            "filtering": {}
        }
        logger.debug("Cleared optimization cache") 