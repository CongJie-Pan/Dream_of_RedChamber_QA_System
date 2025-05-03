"""
Parallel Processing Module for BasicRAG reasoning system.

This module provides parallel processing capabilities to improve the 
performance of retrieval operations when dependencies between sub-questions
allow for concurrent execution.
"""

import time
import asyncio
import concurrent.futures
from typing import Dict, List, Optional, Any, Set, Callable, Awaitable, Union, Tuple
from functools import partial

from .config import logger, AsyncProcessingError, MAX_CONCURRENT_TASKS
from .models import DeepSeekModel

class DependencyGraph:
    """
    Dependency graph for managing execution order of sub-questions.
    
    This class handles the analysis of dependencies between sub-questions
    and determines which ones can be processed in parallel.
    
    Attributes:
        nodes (Dict[int, Dict]): Graph nodes representing sub-questions.
        dependencies (Dict[int, List[int]]): Map of node dependencies.
        reverse_dependencies (Dict[int, List[int]]): Reverse dependency map.
    """
    
    def __init__(self, sub_questions: List[Dict[str, Any]]):
        """
        Initialize the dependency graph from a list of sub-questions.
        
        Args:
            sub_questions (List[Dict[str, Any]]): List of sub-questions with dependencies.
        """
        self.nodes = {}
        self.dependencies = {}
        self.reverse_dependencies = {}
        
        # Build the dependency graph
        for sq in sub_questions:
            sq_id = sq.get("id", 0)
            if sq_id <= 0:
                continue
                
            # Store the node
            self.nodes[sq_id] = sq
            
            # Get dependencies
            deps = sq.get("dependencies", [])
            self.dependencies[sq_id] = deps
            
            # Build reverse dependencies
            for dep_id in deps:
                if dep_id not in self.reverse_dependencies:
                    self.reverse_dependencies[dep_id] = []
                self.reverse_dependencies[dep_id].append(sq_id)
    
    def get_execution_levels(self) -> List[List[int]]:
        """
        Organize nodes into levels for parallel execution.
        
        Returns:
            List[List[int]]: Lists of node IDs grouped by execution level.
        """
        # Create a copy of dependencies to work with
        remaining_deps = {node_id: list(deps) for node_id, deps in self.dependencies.items()}
        
        # Track which nodes have been processed
        processed_nodes = set()
        levels = []
        
        # Continue until all nodes are processed
        while len(processed_nodes) < len(self.nodes):
            # Find nodes with no remaining dependencies
            current_level = []
            for node_id in self.nodes:
                if node_id in processed_nodes:
                    continue
                    
                deps = remaining_deps.get(node_id, [])
                if not deps or all(dep in processed_nodes for dep in deps):
                    current_level.append(node_id)
            
            # If we can't find any nodes to process, we have a cycle
            if not current_level:
                logger.warning("Dependency cycle detected in sub-questions. Breaking cycle.")
                # Find a node with the fewest remaining dependencies and add it
                min_deps = float('inf')
                min_node = None
                for node_id, deps in remaining_deps.items():
                    if node_id not in processed_nodes and len(deps) < min_deps:
                        min_deps = len(deps)
                        min_node = node_id
                
                if min_node:
                    current_level.append(min_node)
                else:
                    # This shouldn't happen, but just in case
                    break
            
            # Add the current level to our results
            levels.append(current_level)
            
            # Mark these nodes as processed
            for node_id in current_level:
                processed_nodes.add(node_id)
                
                # Remove this node from dependencies of other nodes
                for deps in remaining_deps.values():
                    if node_id in deps:
                        deps.remove(node_id)
        
        return levels
    
    def get_max_parallelism(self) -> int:
        """
        Get the maximum level of parallelism possible with this dependency graph.
        
        Returns:
            int: Maximum number of parallel tasks.
        """
        levels = self.get_execution_levels()
        return max(len(level) for level in levels) if levels else 0
    
    def visualize(self) -> Dict[str, Any]:
        """
        Generate a visualization of the dependency graph.
        
        Returns:
            Dict[str, Any]: Data for visualizing the dependency graph.
        """
        levels = self.get_execution_levels()
        
        nodes = []
        for node_id, node in self.nodes.items():
            nodes.append({
                "id": node_id,
                "question": node.get("question", "Unknown"),
                "dependencies": self.dependencies.get(node_id, [])
            })
        
        return {
            "nodes": nodes,
            "levels": levels,
            "max_parallelism": self.get_max_parallelism()
        }

class ParallelProcessor:
    """
    Processor for executing retrieval tasks in parallel when possible.
    
    This class provides methods to optimize the execution of retrieval
    operations by running independent sub-questions concurrently.
    
    Attributes:
        max_workers (int): Maximum number of concurrent workers.
        timeout (float): Timeout for parallel operations.
    """
    
    def __init__(self, max_workers: Optional[int] = None, timeout: float = 30.0):
        """
        Initialize the parallel processor.
        
        Args:
            max_workers (Optional[int]): Maximum number of concurrent workers. If None, uses system defaults.
            timeout (float): Timeout in seconds for parallel operations.
        """
        self.max_workers = max_workers or MAX_CONCURRENT_TASKS
        self.timeout = timeout
        self.executor = None
    
    def process_in_parallel(self, 
                           items: List[Any], 
                           process_func: Callable[[Any], Any]) -> List[Any]:
        """
        Process a list of items in parallel using a processing function.
        
        Args:
            items (List[Any]): List of items to process.
            process_func (Callable[[Any], Any]): Function to process each item.
            
        Returns:
            List[Any]: Results from parallel processing.
        """
        if not items:
            return []
            
        # If only one item, just process it directly
        if len(items) == 1:
            return [process_func(items[0])]
            
        # Use thread pool for CPU-bound tasks or if using blocking I/O
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.max_workers, len(items))) as executor:
            # Submit all tasks
            futures = [executor.submit(process_func, item) for item in items]
            
            # Collect results, respecting timeout
            results = []
            for future in concurrent.futures.as_completed(futures, timeout=self.timeout):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Error in parallel task: {e}")
                    results.append({"error": str(e)})
            
            return results
    
    async def process_async(self, 
                           items: List[Any], 
                           process_func: Callable[[Any], Awaitable[Any]]) -> List[Any]:
        """
        Process a list of items asynchronously.
        
        Args:
            items (List[Any]): List of items to process.
            process_func (Callable[[Any], Awaitable[Any]]): Async function to process each item.
            
        Returns:
            List[Any]: Results from async processing.
        """
        if not items:
            return []
            
        # Create tasks for each item
        tasks = [process_func(item) for item in items]
        
        # Limit concurrency if needed
        if len(tasks) > self.max_workers:
            # Process in batches
            results = []
            for i in range(0, len(tasks), self.max_workers):
                batch = tasks[i:i + self.max_workers]
                batch_results = await asyncio.gather(*batch, return_exceptions=True)
                results.extend(batch_results)
            return results
        else:
            # Process all at once
            return await asyncio.gather(*tasks, return_exceptions=True)
    
    def process_by_levels(self, 
                         sub_questions: List[Dict[str, Any]], 
                         process_func: Callable[[Dict[str, Any]], Any]) -> Dict[str, Any]:
        """
        Process sub-questions in parallelized levels based on their dependencies.
        
        Args:
            sub_questions (List[Dict[str, Any]]): List of sub-questions to process.
            process_func (Callable[[Dict[str, Any]], Any]): Function to process each sub-question.
            
        Returns:
            Dict[str, Any]: Results from processing, keyed by sub-question ID.
        """
        # Create dependency graph
        graph = DependencyGraph(sub_questions)
        
        # Get execution levels
        levels = graph.get_execution_levels()
        logger.info(f"Processing sub-questions in {len(levels)} levels, max parallelism: {graph.get_max_parallelism()}")
        
        # Process each level in sequence, with parallel processing within each level
        all_results = {}
        level_timings = []
        
        for i, level in enumerate(levels):
            logger.info(f"Processing level {i+1} with {len(level)} sub-questions")
            level_start = time.time()
            
            # Get the sub-questions for this level
            level_questions = [graph.nodes.get(sq_id) for sq_id in level if sq_id in graph.nodes]
            
            # Process this level in parallel
            level_results = self.process_in_parallel(level_questions, process_func)
            
            # Store results
            for j, sq_id in enumerate(level):
                if j < len(level_results):
                    all_results[str(sq_id)] = level_results[j]
            
            level_time = time.time() - level_start
            level_timings.append({
                "level": i+1,
                "node_count": len(level),
                "processing_time": level_time
            })
            logger.info(f"Level {i+1} completed in {level_time:.2f}s")
        
        # Return all results and timing information
        return {
            "results": all_results,
            "level_timings": level_timings,
            "total_levels": len(levels),
            "max_parallelism": graph.get_max_parallelism()
        }
        
    async def process_by_levels_async(self, 
                                    sub_questions: List[Dict[str, Any]], 
                                    process_func: Callable[[Dict[str, Any]], Awaitable[Any]]) -> Dict[str, Any]:
        """
        Process sub-questions in parallelized levels asynchronously based on their dependencies.
        
        Args:
            sub_questions (List[Dict[str, Any]]): List of sub-questions to process.
            process_func (Callable[[Dict[str, Any]], Awaitable[Any]]): Async function to process each sub-question.
            
        Returns:
            Dict[str, Any]: Results from processing, keyed by sub-question ID.
        """
        # Create dependency graph
        graph = DependencyGraph(sub_questions)
        
        # Get execution levels
        levels = graph.get_execution_levels()
        logger.info(f"Processing sub-questions asynchronously in {len(levels)} levels, max parallelism: {graph.get_max_parallelism()}")
        
        # Process each level in sequence, with parallel processing within each level
        all_results = {}
        level_timings = []
        
        for i, level in enumerate(levels):
            logger.info(f"Processing level {i+1} with {len(level)} sub-questions")
            level_start = time.time()
            
            # Get the sub-questions for this level
            level_questions = [graph.nodes.get(sq_id) for sq_id in level if sq_id in graph.nodes]
            
            # Process this level in parallel
            level_results = await self.process_async(level_questions, process_func)
            
            # Store results
            for j, sq_id in enumerate(level):
                if j < len(level_results):
                    # Check for exceptions
                    result = level_results[j]
                    if isinstance(result, Exception):
                        logger.error(f"Error processing sub-question {sq_id}: {result}")
                        all_results[str(sq_id)] = {"error": str(result)}
                    else:
                        all_results[str(sq_id)] = result
            
            level_time = time.time() - level_start
            level_timings.append({
                "level": i+1,
                "node_count": len(level),
                "processing_time": level_time
            })
            logger.info(f"Level {i+1} completed in {level_time:.2f}s")
        
        # Return all results and timing information
        return {
            "results": all_results,
            "level_timings": level_timings,
            "total_levels": len(levels),
            "max_parallelism": graph.get_max_parallelism()
        }

class RetrievalOrchestrator:
    """
    Orchestrator for managing retrieval operations, with support for parallel processing.
    
    This class serves as a higher-level interface for the reasoning pipeline
    to implement parallel processing of retrieval operations.
    
    Attributes:
        processor (ParallelProcessor): Parallel processor for retrieval operations.
        model (DeepSeekModel): Language model for reasoning operations.
    """
    
    def __init__(self, model: Optional[DeepSeekModel] = None, max_workers: Optional[int] = None):
        """
        Initialize the retrieval orchestrator.
        
        Args:
            model (Optional[DeepSeekModel]): Language model to use. If None, creates a new instance.
            max_workers (Optional[int]): Maximum number of concurrent workers.
        """
        self.processor = ParallelProcessor(max_workers=max_workers)
        self.model = model or DeepSeekModel()
    
    def optimize_execution_plan(self, 
                               sub_questions: List[Dict[str, Any]], 
                               query_complexity: Optional[str] = None) -> Dict[str, Any]:
        """
        Create an optimized execution plan for processing sub-questions.
        
        Args:
            sub_questions (List[Dict[str, Any]]): List of sub-questions to process.
            query_complexity (Optional[str]): Query complexity level.
            
        Returns:
            Dict[str, Any]: Optimized execution plan.
        """
        # Create dependency graph
        graph = DependencyGraph(sub_questions)
        
        # Get execution levels for parallelism
        levels = graph.get_execution_levels()
        max_parallelism = graph.get_max_parallelism()
        
        # Determine if parallel processing is worth it
        # BasicRAG is generally more suitable for parallelization
        # since it only uses vector search without hybrid methods
        use_parallel = max_parallelism > 1 and len(sub_questions) > 1
        
        # For BasicRAG, we can be more aggressive with parallelization
        # since we're only using vector search
        if query_complexity == "complex" and len(sub_questions) > 5:
            # For very complex queries, still ensure we have enough parallelism to benefit
            use_parallel = max_parallelism >= 2
        
        # Create the plan
        plan = {
            "sub_question_count": len(sub_questions),
            "execution_levels": levels,
            "max_parallelism": max_parallelism,
            "use_parallel": use_parallel,
            "visualization": graph.visualize()
        }
        
        return plan
    
    def execute_with_plan(self, 
                         sub_questions: List[Dict[str, Any]], 
                         process_func: Callable[[Dict[str, Any]], Any],
                         parallel: Optional[bool] = None) -> Dict[str, Any]:
        """
        Execute retrieval operations according to an optimized plan.
        
        Args:
            sub_questions (List[Dict[str, Any]]): List of sub-questions to process.
            process_func (Callable[[Dict[str, Any]], Any]): Function to process each sub-question.
            parallel (Optional[bool]): Whether to use parallel processing. If None, decides automatically.
            
        Returns:
            Dict[str, Any]: Results from processing.
        """
        # First create a plan if we need to decide on parallelism
        if parallel is None:
            plan = self.optimize_execution_plan(sub_questions)
            parallel = plan.get("use_parallel", False)
        
        # If parallel processing is enabled, use level-based processing
        if parallel:
            start_time = time.time()
            results = self.processor.process_by_levels(sub_questions, process_func)
            total_time = time.time() - start_time
            
            logger.info(f"Parallel processing completed in {total_time:.2f}s")
            results["total_time"] = total_time
            results["parallel_enabled"] = True
            
            return results
        else:
            # Process sequentially
            start_time = time.time()
            all_results = {}
            
            for sq in sub_questions:
                sq_id = sq.get("id", 0)
                try:
                    result = process_func(sq)
                    all_results[str(sq_id)] = result
                except Exception as e:
                    logger.error(f"Error processing sub-question {sq_id}: {e}")
                    all_results[str(sq_id)] = {"error": str(e)}
            
            total_time = time.time() - start_time
            logger.info(f"Sequential processing completed in {total_time:.2f}s")
            
            return {
                "results": all_results,
                "total_time": total_time,
                "parallel_enabled": False
            }
    
    async def execute_with_plan_async(self, 
                                    sub_questions: List[Dict[str, Any]], 
                                    process_func: Callable[[Dict[str, Any]], Awaitable[Any]],
                                    parallel: Optional[bool] = None) -> Dict[str, Any]:
        """
        Execute retrieval operations asynchronously according to an optimized plan.
        
        Args:
            sub_questions (List[Dict[str, Any]]): List of sub-questions to process.
            process_func (Callable[[Dict[str, Any]], Awaitable[Any]]): Async function to process each sub-question.
            parallel (Optional[bool]): Whether to use parallel processing. If None, decides automatically.
            
        Returns:
            Dict[str, Any]: Results from processing.
        """
        # First create a plan if we need to decide on parallelism
        if parallel is None:
            plan = self.optimize_execution_plan(sub_questions)
            parallel = plan.get("use_parallel", False)
        
        # If parallel processing is enabled, use level-based processing
        if parallel:
            start_time = time.time()
            try:
                results = await self.processor.process_by_levels_async(sub_questions, process_func)
                total_time = time.time() - start_time
                
                logger.info(f"Parallel async processing completed in {total_time:.2f}s")
                results["total_time"] = total_time
                results["parallel_enabled"] = True
                
                return results
            except Exception as e:
                logger.error(f"Error in parallel async processing: {e}")
                # Fall back to sequential processing
                return await self._process_sequential_async(sub_questions, process_func)
        else:
            # Process sequentially
            return await self._process_sequential_async(sub_questions, process_func)
    
    async def _process_sequential_async(self, 
                                      sub_questions: List[Dict[str, Any]], 
                                      process_func: Callable[[Dict[str, Any]], Awaitable[Any]]) -> Dict[str, Any]:
        """
        Process sub-questions sequentially using async functions.
        
        Args:
            sub_questions (List[Dict[str, Any]]): List of sub-questions to process.
            process_func (Callable[[Dict[str, Any]], Awaitable[Any]]): Async function to process each sub-question.
            
        Returns:
            Dict[str, Any]: Results from processing.
        """
        start_time = time.time()
        all_results = {}
        
        for sq in sub_questions:
            sq_id = sq.get("id", 0)
            try:
                result = await process_func(sq)
                all_results[str(sq_id)] = result
            except Exception as e:
                logger.error(f"Error processing sub-question {sq_id}: {e}")
                all_results[str(sq_id)] = {"error": str(e)}
        
        total_time = time.time() - start_time
        logger.info(f"Sequential async processing completed in {total_time:.2f}s")
        
        return {
            "results": all_results,
            "total_time": total_time,
            "parallel_enabled": False
        } 