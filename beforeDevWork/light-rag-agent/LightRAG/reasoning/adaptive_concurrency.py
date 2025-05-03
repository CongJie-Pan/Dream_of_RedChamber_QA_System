"""
Adaptive Concurrency Control Module

This module provides adaptive concurrency control functionality
to dynamically adjust the number of concurrent tasks based on
system load and resource availability.

Features:
- Dynamic concurrency adjustment based on system load
- CPU and memory utilization monitoring
- API rate limit tracking
- Performance trend analysis
"""

import os
import time
import threading
import logging
import psutil
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from .config import logger, MAX_CONCURRENT_TASKS


@dataclass
class SystemMetrics:
    """
    Class to store system performance metrics.
    
    This dataclass represents a snapshot of system metrics at a point in time,
    including CPU usage, memory usage, and disk I/O.
    
    Attributes:
        timestamp (float): Time when metrics were captured
        cpu_percent (float): CPU utilization percentage
        memory_percent (float): Memory utilization percentage
        io_counters (Dict[str, int]): Disk I/O counters
        thread_count (int): Number of active threads
        process_count (int): Number of active processes
    """
    timestamp: float = field(default_factory=time.time)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    io_counters: Dict[str, int] = field(default_factory=dict)
    thread_count: int = 0
    process_count: int = 0
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            "timestamp": self.timestamp,
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "io_counters": self.io_counters,
            "thread_count": self.thread_count,
            "process_count": self.process_count
        }


@dataclass
class ApiMetrics:
    """
    Class to store API usage metrics.
    
    This dataclass tracks API usage patterns, rate limits, and failures
    to inform concurrency decisions for API-dependent operations.
    
    Attributes:
        timestamp (float): Time when metrics were captured
        total_calls (int): Total number of API calls
        successful_calls (int): Number of successful API calls
        failed_calls (int): Number of failed API calls
        rate_limit_hits (int): Number of rate limit errors
        average_latency (float): Average latency in seconds
        token_usage (int): Token usage count
    """
    timestamp: float = field(default_factory=time.time)
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rate_limit_hits: int = 0
    average_latency: float = 0.0
    token_usage: int = 0
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            "timestamp": self.timestamp,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "rate_limit_hits": self.rate_limit_hits,
            "average_latency": self.average_latency,
            "token_usage": self.token_usage
        }
    
    @property
    def success_rate(self) -> float:
        """Calculate the API call success rate."""
        if self.total_calls == 0:
            return 1.0  # Default to 100% if no calls
        return self.successful_calls / self.total_calls


class AdaptiveConcurrencyManager:
    """
    Manager for adaptive concurrency control.
    
    This class monitors system load and API usage metrics to dynamically
    adjust the concurrency level for optimal performance.
    
    Attributes:
        max_concurrent_tasks (int): Maximum allowed concurrent tasks
        current_concurrent_tasks (int): Current concurrency setting
        monitoring_interval (float): Interval for system monitoring in seconds
        system_metrics_history (deque): Recent system metrics history
        api_metrics_history (deque): Recent API metrics history
        monitoring_active (bool): Whether monitoring is active
        monitor_thread (threading.Thread): Thread for continuous monitoring
        high_load_threshold (float): CPU usage threshold for high load
        critical_load_threshold (float): CPU usage threshold for critical load
        rate_limit_window (int): Time window for rate limit tracking in seconds
    """
    
    def __init__(self, initial_concurrent_tasks: Optional[int] = None,
               max_concurrent_tasks: Optional[int] = None,
               monitoring_interval: float = 5.0,
               metrics_history_size: int = 60):
        """
        Initialize the adaptive concurrency manager.
        
        Args:
            initial_concurrent_tasks (Optional[int]): Initial concurrency level
            max_concurrent_tasks (Optional[int]): Maximum allowed concurrent tasks
            monitoring_interval (float): Interval for system monitoring in seconds
            metrics_history_size (int): Number of metrics snapshots to keep in history
        """
        # Initialize concurrency settings
        self.max_concurrent_tasks = max_concurrent_tasks or MAX_CONCURRENT_TASKS
        self.current_concurrent_tasks = initial_concurrent_tasks or max(1, self.max_concurrent_tasks // 2)
        
        # Initialize monitoring settings
        self.monitoring_interval = monitoring_interval
        self.system_metrics_history = deque(maxlen=metrics_history_size)
        self.api_metrics_history = deque(maxlen=metrics_history_size)
        
        # Thresholds for adjustment decisions
        self.high_load_threshold = 80.0  # CPU usage percentage
        self.critical_load_threshold = 90.0  # CPU usage percentage
        self.memory_warning_threshold = 85.0  # Memory usage percentage
        self.rate_limit_window = 300  # 5 minutes in seconds
        
        # Thread control for monitoring
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Current API metrics for the current window
        self.current_api_metrics = ApiMetrics()
        self.api_call_times = deque(maxlen=100)  # Track recent call timestamps
        
        logger.info(f"AdaptiveConcurrencyManager initialized with max_tasks={self.max_concurrent_tasks}, "
                   f"initial_tasks={self.current_concurrent_tasks}")
    
    def start_monitoring(self) -> None:
        """
        Start continuous monitoring of system metrics.
        
        This method starts a background thread that periodically
        collects system metrics and adjusts concurrency accordingly.
        """
        if self.monitoring_active:
            logger.warning("Monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("System metrics monitoring started")
    
    def stop_monitoring(self) -> None:
        """
        Stop the continuous monitoring of system metrics.
        """
        if not self.monitoring_active:
            logger.warning("Monitoring is not active")
            return
        
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("System metrics monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """
        Background loop for continuous system monitoring.
        
        This method runs in a separate thread and periodically
        collects system metrics and adjusts concurrency.
        """
        while self.monitoring_active:
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()
                
                # Store metrics in history
                with self.lock:
                    self.system_metrics_history.append(metrics)
                
                # Adjust concurrency based on metrics
                self._adjust_concurrency(metrics)
                
                # Wait for the next monitoring interval
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)  # Continue monitoring despite errors
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """
        Collect current system performance metrics.
        
        Returns:
            SystemMetrics: Current system metrics
        """
        try:
            # Get CPU usage (averaged across all cores)
            cpu_percent = psutil.cpu_percent(interval=0.5)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Get disk I/O counters
            io = psutil.disk_io_counters()
            io_counters = {
                "read_count": io.read_count if io else 0,
                "write_count": io.write_count if io else 0,
                "read_bytes": io.read_bytes if io else 0,
                "write_bytes": io.write_bytes if io else 0
            }
            
            # Get thread and process counts
            current_process = psutil.Process()
            thread_count = len(current_process.threads())
            process_count = len(psutil.pids())
            
            # Create and return metrics
            metrics = SystemMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                io_counters=io_counters,
                thread_count=thread_count,
                process_count=process_count
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            # Return default metrics on error
            return SystemMetrics(timestamp=time.time())
    
    def _adjust_concurrency(self, metrics: SystemMetrics) -> None:
        """
        Adjust concurrency level based on current system metrics.
        
        This method implements the adaptive algorithm that determines
        the optimal concurrency level based on system load and API usage.
        
        Args:
            metrics (SystemMetrics): Current system metrics
        """
        with self.lock:
            previous_concurrency = self.current_concurrent_tasks
            
            # Check for critical load conditions
            if metrics.cpu_percent >= self.critical_load_threshold:
                # Critical load: reduce concurrency significantly
                self.current_concurrent_tasks = max(1, self.current_concurrent_tasks // 2)
                logger.warning(f"Critical CPU load detected ({metrics.cpu_percent:.1f}%). "
                              f"Reducing concurrency to {self.current_concurrent_tasks}")
                
            elif metrics.memory_percent >= self.memory_warning_threshold:
                # High memory usage: reduce concurrency moderately
                self.current_concurrent_tasks = max(1, int(self.current_concurrent_tasks * 0.75))
                logger.warning(f"High memory usage detected ({metrics.memory_percent:.1f}%). "
                              f"Reducing concurrency to {self.current_concurrent_tasks}")
                
            elif metrics.cpu_percent >= self.high_load_threshold:
                # High load: reduce concurrency slightly
                self.current_concurrent_tasks = max(1, self.current_concurrent_tasks - 1)
                logger.info(f"High CPU load detected ({metrics.cpu_percent:.1f}%). "
                           f"Reducing concurrency to {self.current_concurrent_tasks}")
                
            else:
                # Check recent API metrics for rate limit issues
                recent_rate_limits = self._count_recent_rate_limits()
                
                if recent_rate_limits > 0:
                    # Rate limit issues: reduce concurrency
                    self.current_concurrent_tasks = max(1, self.current_concurrent_tasks - 1)
                    logger.info(f"API rate limits detected ({recent_rate_limits} hits). "
                               f"Reducing concurrency to {self.current_concurrent_tasks}")
                    
                elif metrics.cpu_percent < 50.0 and self.current_concurrent_tasks < self.max_concurrent_tasks:
                    # Low load: consider increasing concurrency
                    # Use a gradual increase approach
                    if len(self.system_metrics_history) >= 3:
                        # Check if load has been consistently low
                        recent_metrics = list(self.system_metrics_history)[-3:]
                        avg_recent_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
                        
                        if avg_recent_cpu < 50.0:
                            # Increase concurrency gradually
                            self.current_concurrent_tasks = min(
                                self.max_concurrent_tasks,
                                self.current_concurrent_tasks + 1
                            )
                            logger.info(f"Low CPU load detected ({avg_recent_cpu:.1f}%). "
                                       f"Increasing concurrency to {self.current_concurrent_tasks}")
            
            # Log changes in concurrency
            if self.current_concurrent_tasks != previous_concurrency:
                logger.info(f"Concurrency adjusted from {previous_concurrency} to {self.current_concurrent_tasks}")
    
    def _count_recent_rate_limits(self) -> int:
        """
        Count recent API rate limit hits.
        
        Returns:
            int: Number of rate limit hits in the recent window
        """
        # Get API metrics from the recent window
        window_start = time.time() - self.rate_limit_window
        recent_metrics = [m for m in self.api_metrics_history if m.timestamp >= window_start]
        
        # Count rate limit hits
        rate_limit_hits = sum(m.rate_limit_hits for m in recent_metrics)
        
        return rate_limit_hits
    
    def get_concurrent_tasks(self) -> int:
        """
        Get the current recommended concurrency level.
        
        Returns:
            int: Current recommended number of concurrent tasks
        """
        with self.lock:
            return self.current_concurrent_tasks
    
    def record_api_call(self, success: bool, latency: float,
                      rate_limited: bool = False, token_count: int = 0) -> None:
        """
        Record an API call for metrics tracking.
        
        Args:
            success (bool): Whether the API call was successful
            latency (float): Latency of the API call in seconds
            rate_limited (bool): Whether the call hit a rate limit
            token_count (int): Number of tokens used in the call
        """
        with self.lock:
            # Update current API metrics
            self.current_api_metrics.total_calls += 1
            
            if success:
                self.current_api_metrics.successful_calls += 1
            else:
                self.current_api_metrics.failed_calls += 1
            
            if rate_limited:
                self.current_api_metrics.rate_limit_hits += 1
            
            # Update average latency
            if self.current_api_metrics.average_latency == 0:
                self.current_api_metrics.average_latency = latency
            else:
                # Running average calculation
                prev_avg = self.current_api_metrics.average_latency
                prev_count = self.current_api_metrics.total_calls - 1
                self.current_api_metrics.average_latency = (prev_avg * prev_count + latency) / self.current_api_metrics.total_calls
            
            # Update token usage
            self.current_api_metrics.token_usage += token_count
            
            # Record call timestamp for rate analysis
            self.api_call_times.append(time.time())
            
            # Periodically snapshot API metrics (every 10 calls)
            if self.current_api_metrics.total_calls % 10 == 0:
                self.api_metrics_history.append(self.current_api_metrics)
                self.current_api_metrics = ApiMetrics()  # Reset for next window
    
    def get_current_rate(self) -> float:
        """
        Calculate the current API call rate per minute.
        
        Returns:
            float: Current API calls per minute
        """
        with self.lock:
            if not self.api_call_times:
                return 0.0
            
            # Get calls in the last minute
            minute_ago = time.time() - 60
            recent_calls = [t for t in self.api_call_times if t >= minute_ago]
            
            return len(recent_calls)
    
    def get_system_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of recent system metrics.
        
        Returns:
            Dict[str, Any]: Summary of system metrics
        """
        with self.lock:
            if not self.system_metrics_history:
                return {"status": "No metrics available"}
            
            # Get most recent metrics
            latest = self.system_metrics_history[-1]
            
            # Calculate averages for the last minute
            minute_ago = time.time() - 60
            recent_metrics = [m for m in self.system_metrics_history if m.timestamp >= minute_ago]
            
            if recent_metrics:
                avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
                avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
                avg_threads = sum(m.thread_count for m in recent_metrics) / len(recent_metrics)
            else:
                avg_cpu = latest.cpu_percent
                avg_memory = latest.memory_percent
                avg_threads = latest.thread_count
            
            return {
                "current": {
                    "timestamp": datetime.fromtimestamp(latest.timestamp).isoformat(),
                    "cpu_percent": latest.cpu_percent,
                    "memory_percent": latest.memory_percent,
                    "thread_count": latest.thread_count
                },
                "average_last_minute": {
                    "cpu_percent": avg_cpu,
                    "memory_percent": avg_memory,
                    "thread_count": avg_threads
                },
                "concurrency": {
                    "current": self.current_concurrent_tasks,
                    "maximum": self.max_concurrent_tasks
                }
            }
    
    def get_api_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of recent API metrics.
        
        Returns:
            Dict[str, Any]: Summary of API metrics
        """
        with self.lock:
            # Combine current metrics with history
            all_metrics = list(self.api_metrics_history)
            
            if not all_metrics:
                if self.current_api_metrics.total_calls > 0:
                    all_metrics = [self.current_api_metrics]
                else:
                    return {"status": "No API metrics available"}
            
            # Calculate totals
            total_calls = sum(m.total_calls for m in all_metrics)
            successful_calls = sum(m.successful_calls for m in all_metrics)
            failed_calls = sum(m.failed_calls for m in all_metrics)
            rate_limits = sum(m.rate_limit_hits for m in all_metrics)
            
            # Calculate overall success rate
            success_rate = successful_calls / total_calls if total_calls > 0 else 1.0
            
            # Calculate average latency (weighted by call count)
            if total_calls > 0:
                weighted_latencies = sum(m.average_latency * m.total_calls for m in all_metrics)
                avg_latency = weighted_latencies / total_calls
            else:
                avg_latency = 0.0
            
            # Get current rate
            current_rate = self.get_current_rate()
            
            return {
                "total_calls": total_calls,
                "successful_calls": successful_calls,
                "failed_calls": failed_calls,
                "rate_limit_hits": rate_limits,
                "success_rate": success_rate,
                "average_latency": avg_latency,
                "current_rate_per_minute": current_rate,
                "token_usage": sum(m.token_usage for m in all_metrics)
            }
    
    def recommend_batch_size(self, max_size: int = 10) -> int:
        """
        Recommend a batch size for API calls based on current system state.
        
        Args:
            max_size (int): Maximum batch size to consider
            
        Returns:
            int: Recommended batch size
        """
        with self.lock:
            # Start with current concurrency level
            base_size = min(self.current_concurrent_tasks, max_size)
            
            # Adjust based on recent API metrics
            api_summary = self.get_api_metrics_summary()
            
            # If we've had rate limit issues, reduce batch size
            if "rate_limit_hits" in api_summary and api_summary["rate_limit_hits"] > 0:
                base_size = max(1, base_size - 1)
            
            # If success rate is low, reduce batch size
            if "success_rate" in api_summary and api_summary["success_rate"] < 0.9:
                base_size = max(1, base_size - 1)
            
            # If current rate is high, consider reducing batch size
            if "current_rate_per_minute" in api_summary and api_summary["current_rate_per_minute"] > 30:
                base_size = max(1, base_size - 1)
            
            return base_size
    
    def set_max_concurrent_tasks(self, max_tasks: int) -> None:
        """
        Set the maximum allowed concurrent tasks.
        
        Args:
            max_tasks (int): New maximum concurrent tasks
        """
        with self.lock:
            if max_tasks < 1:
                logger.warning(f"Invalid max_tasks value: {max_tasks}. Using 1.")
                max_tasks = 1
            
            self.max_concurrent_tasks = max_tasks
            
            # Adjust current concurrency if it exceeds the new maximum
            if self.current_concurrent_tasks > max_tasks:
                self.current_concurrent_tasks = max_tasks
                
            logger.info(f"Maximum concurrent tasks set to {max_tasks}")
    
    def get_metrics_history(self, 
                         metrics_type: str = "system",
                         limit: int = 60) -> List[Dict[str, Any]]:
        """
        Get history of metrics for analysis.
        
        Args:
            metrics_type (str): Type of metrics to return ("system" or "api")
            limit (int): Maximum number of metrics to return
            
        Returns:
            List[Dict[str, Any]]: List of metrics in dictionary format
        """
        with self.lock:
            if metrics_type == "system":
                metrics = list(self.system_metrics_history)
                return [m.as_dict() for m in metrics[-limit:]]
            elif metrics_type == "api":
                metrics = list(self.api_metrics_history)
                return [m.as_dict() for m in metrics[-limit:]]
            else:
                logger.warning(f"Unknown metrics type: {metrics_type}")
                return []


class AdaptiveBatchProcessor:
    """
    Processor for adaptively batching API calls based on system load.
    
    This class provides functionality for adaptively batching API calls
    based on system load, API rate limits, and performance trends.
    
    Attributes:
        concurrency_manager (AdaptiveConcurrencyManager): Concurrency manager
        max_batch_size (int): Maximum batch size
        token_counter (Callable): Function for counting tokens
    """
    
    def __init__(self, concurrency_manager: AdaptiveConcurrencyManager,
               max_batch_size: int = 10,
               token_counter: Optional[Callable[[str], int]] = None):
        """
        Initialize the adaptive batch processor.
        
        Args:
            concurrency_manager (AdaptiveConcurrencyManager): Concurrency manager
            max_batch_size (int): Maximum batch size
            token_counter (Optional[Callable[[str], int]]): Function for counting tokens
        """
        self.concurrency_manager = concurrency_manager
        self.max_batch_size = max_batch_size
        self.token_counter = token_counter
        
        # Start monitoring if not already started
        if not concurrency_manager.monitoring_active:
            concurrency_manager.start_monitoring()
    
    def batch_items(self, items: List[Any]) -> List[List[Any]]:
        """
        Split items into adaptive batches based on current system state.
        
        Args:
            items (List[Any]): Items to batch
            
        Returns:
            List[List[Any]]: Items split into batches
        """
        if not items:
            return []
        
        # Get recommended batch size
        batch_size = self.concurrency_manager.recommend_batch_size(self.max_batch_size)
        
        # Create batches
        batches = []
        for i in range(0, len(items), batch_size):
            batches.append(items[i:i + batch_size])
        
        logger.debug(f"Split {len(items)} items into {len(batches)} batches of size {batch_size}")
        return batches
    
    def batch_by_tokens(self, items: List[Tuple[str, Any]], 
                     max_tokens_per_batch: int = 4000) -> List[List[Tuple[str, Any]]]:
        """
        Batch items based on token count and system state.
        
        Args:
            items (List[Tuple[str, Any]]): Items to batch with text content
            max_tokens_per_batch (int): Maximum tokens per batch
            
        Returns:
            List[List[Tuple[str, Any]]]: Items split into batches
        """
        if not self.token_counter:
            logger.warning("No token counter provided, falling back to regular batching")
            return self.batch_items(items)
        
        if not items:
            return []
        
        # Get recommended batch size based on system state
        recommended_batch_size = self.concurrency_manager.recommend_batch_size(self.max_batch_size)
        
        # Create batches based on token count
        batches = []
        current_batch = []
        current_tokens = 0
        
        for item_text, item_data in items:
            # Count tokens in this item
            item_tokens = self.token_counter(item_text)
            
            # If adding this item would exceed the token limit or batch size,
            # start a new batch (unless the current batch is empty)
            if (current_batch and 
                (current_tokens + item_tokens > max_tokens_per_batch or 
                 len(current_batch) >= recommended_batch_size)):
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            
            # Add item to current batch
            current_batch.append((item_text, item_data))
            current_tokens += item_tokens
            
            # If this single item is too large for a batch, log a warning
            if item_tokens > max_tokens_per_batch:
                logger.warning(f"Item with {item_tokens} tokens exceeds max_tokens_per_batch ({max_tokens_per_batch})")
        
        # Add the last batch if not empty
        if current_batch:
            batches.append(current_batch)
        
        logger.debug(f"Split {len(items)} items into {len(batches)} token-aware batches")
        return batches
    
    def process_batches(self, 
                      batches: List[List[Any]], 
                      process_func: Callable[[List[Any]], List[Any]],
                      record_metrics: bool = True) -> List[Any]:
        """
        Process batches and collect results.
        
        Args:
            batches (List[List[Any]]): Batches to process
            process_func (Callable[[List[Any]], List[Any]]): Function to process each batch
            record_metrics (bool): Whether to record processing metrics
            
        Returns:
            List[Any]: Combined results from all batches
        """
        if not batches:
            return []
        
        all_results = []
        
        for i, batch in enumerate(batches):
            try:
                logger.debug(f"Processing batch {i+1}/{len(batches)} with {len(batch)} items")
                
                start_time = time.time()
                batch_results = process_func(batch)
                end_time = time.time()
                
                latency = end_time - start_time
                success = True
                rate_limited = False
                
                all_results.extend(batch_results)
                
                if record_metrics:
                    self.concurrency_manager.record_api_call(
                        success=success,
                        latency=latency,
                        rate_limited=rate_limited,
                        token_count=0  # Placeholder, replace with actual count if available
                    )
                
            except Exception as e:
                logger.error(f"Error processing batch {i+1}: {e}")
                
                # Check for rate limit errors
                rate_limited = "rate limit" in str(e).lower() or "too many requests" in str(e).lower()
                
                if record_metrics:
                    self.concurrency_manager.record_api_call(
                        success=False,
                        latency=0.0,
                        rate_limited=rate_limited,
                        token_count=0
                    )
                
                # If rate limited, add delay before continuing
                if rate_limited:
                    logger.warning(f"Rate limit detected, adding delay before next batch")
                    time.sleep(5.0)  # 5-second backoff
        
        return all_results
    
    async def process_batches_async(self, 
                                  batches: List[List[Any]], 
                                  process_func_async: Callable[[List[Any]], Any],
                                  record_metrics: bool = True) -> List[Any]:
        """
        Process batches asynchronously and collect results.
        
        Args:
            batches (List[List[Any]]): Batches to process
            process_func_async (Callable[[List[Any]], Any]): Async function to process each batch
            record_metrics (bool): Whether to record processing metrics
            
        Returns:
            List[Any]: Combined results from all batches
        """
        if not batches:
            return []
        
        import asyncio
        all_results = []
        
        for i, batch in enumerate(batches):
            try:
                logger.debug(f"Processing batch {i+1}/{len(batches)} with {len(batch)} items")
                
                start_time = time.time()
                batch_results = await process_func_async(batch)
                end_time = time.time()
                
                latency = end_time - start_time
                success = True
                rate_limited = False
                
                all_results.extend(batch_results)
                
                if record_metrics:
                    self.concurrency_manager.record_api_call(
                        success=success,
                        latency=latency,
                        rate_limited=rate_limited,
                        token_count=0
                    )
                
            except Exception as e:
                logger.error(f"Error processing batch {i+1}: {e}")
                
                # Check for rate limit errors
                rate_limited = "rate limit" in str(e).lower() or "too many requests" in str(e).lower()
                
                if record_metrics:
                    self.concurrency_manager.record_api_call(
                        success=False,
                        latency=0.0,
                        rate_limited=rate_limited,
                        token_count=0
                    )
                
                # If rate limited, add delay before continuing
                if rate_limited:
                    logger.warning(f"Rate limit detected, adding delay before next batch")
                    await asyncio.sleep(5.0)  # 5-second backoff
        
        return all_results
    
    def shutdown(self) -> None:
        """Stop monitoring and clean up resources."""
        self.concurrency_manager.stop_monitoring()
        logger.info("AdaptiveBatchProcessor shutdown complete") 