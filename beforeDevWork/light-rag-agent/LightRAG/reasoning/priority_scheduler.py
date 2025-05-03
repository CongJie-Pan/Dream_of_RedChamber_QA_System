"""
Priority Scheduler for Time-Sensitive Questions

This module provides priority-based scheduling functionality for
processing time-sensitive questions and tasks with different
priority levels.

Features:
- Multiple priority levels for tasks
- Deadline-aware scheduling
- Dynamic priority adjustment
- Resource allocation based on priority
"""

import time
import heapq
import threading
from typing import Dict, List, Any, Callable, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from .config import logger


class Priority(Enum):
    """
    Priority levels for scheduled tasks.
    
    The lower the numeric value, the higher the priority.
    """
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    BACKGROUND = 4


@dataclass(order=True)
class PrioritizedTask:
    """
    A task with priority information for scheduling.
    
    Attributes:
        priority (int): Task priority level
        deadline (float): Task deadline timestamp (optional)
        create_time (float): Task creation time
        task_id (str): Unique task identifier
        name (str): Human-readable task name
        data (Dict[str, Any]): Task data
        execute_func (Callable): Function to execute for this task
        is_cancelled (bool): Whether the task is cancelled
    """
    # Fields used for sorting (in order)
    priority: int
    deadline: float = field(default=0.0)
    create_time: float = field(default_factory=time.time)
    
    # Other fields (not used for comparison)
    task_id: str = field(default="", compare=False)
    name: str = field(default="", compare=False)
    data: Dict[str, Any] = field(default_factory=dict, compare=False)
    execute_func: Callable = field(default=None, compare=False)
    is_cancelled: bool = field(default=False, compare=False)
    
    def __post_init__(self):
        """Post-initialization setup for tasks with no deadline."""
        # If deadline is not specified, set a default far future deadline
        if self.deadline == 0.0:
            # Set a default deadline based on priority
            # Higher priority = longer deadline by default
            days_to_add = {
                Priority.CRITICAL.value: 0.1,  # 2.4 hours
                Priority.HIGH.value: 1,     # 1 day
                Priority.MEDIUM.value: 3,   # 3 days
                Priority.LOW.value: 7,      # 7 days
                Priority.BACKGROUND.value: 30  # 30 days
            }.get(self.priority, 1)
            
            self.deadline = time.time() + (days_to_add * 86400)  # days in seconds


class PriorityScheduler:
    """
    Scheduler for prioritizing and scheduling tasks.
    
    This class manages task priorities, deadlines, and execution,
    ensuring that high-priority and time-sensitive tasks are
    processed before less critical ones.
    
    Attributes:
        task_queue (List[PrioritizedTask]): Priority queue of tasks
        active_tasks (Dict[str, PrioritizedTask]): Currently active tasks
        completed_tasks (Dict[str, Dict[str, Any]]): Completed task results
        lock (threading.RLock): Lock for thread safety
        worker_thread (threading.Thread): Background worker thread
        shutdown_flag (bool): Flag to indicate shutdown
    """
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize the priority scheduler.
        
        Args:
            max_workers (int): Maximum number of worker threads
        """
        self.task_queue = []  # Priority queue (heap)
        self.active_tasks = {}  # Tasks currently being processed
        self.completed_tasks = {}  # Results of completed tasks
        self.lock = threading.RLock()  # Reentrant lock for thread safety
        self.condition = threading.Condition(self.lock)  # Condition for waiting
        self.worker_thread = None
        self.shutdown_flag = False
        self.max_workers = max_workers
        self.current_workers = 0
        
        logger.info(f"PriorityScheduler initialized with max_workers={max_workers}")
    
    def start(self) -> None:
        """Start the worker thread for processing tasks."""
        with self.lock:
            if self.worker_thread is not None:
                return  # Already started
            
            self.shutdown_flag = False
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            logger.info("PriorityScheduler worker thread started")
    
    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the scheduler.
        
        Args:
            wait (bool): Whether to wait for tasks to complete
        """
        with self.lock:
            if self.worker_thread is None:
                return  # Already stopped
            
            self.shutdown_flag = True
            self.condition.notify_all()  # Wake up any waiting threads
        
        if wait and self.worker_thread:
            self.worker_thread.join()
            self.worker_thread = None
            logger.info("PriorityScheduler worker thread shut down")
    
    def _worker_loop(self) -> None:
        """Worker thread loop for processing tasks."""
        while not self.shutdown_flag:
            task = None
            
            # Get the next task from the queue
            with self.lock:
                # Remove cancelled tasks from the queue
                while self.task_queue and self.task_queue[0].is_cancelled:
                    heapq.heappop(self.task_queue)
                
                # If no tasks or all workers busy, wait for a change
                if not self.task_queue or self.current_workers >= self.max_workers:
                    self.condition.wait(timeout=1.0)  # Wait with timeout
                    continue
                
                # Get the highest priority task
                if self.task_queue:
                    task = heapq.heappop(self.task_queue)
                    self.active_tasks[task.task_id] = task
                    self.current_workers += 1
            
            # Process the task outside of the lock
            if task and not task.is_cancelled:
                self._process_task(task)
            
            # Short sleep to prevent CPU spinning
            time.sleep(0.01)
    
    def _process_task(self, task: PrioritizedTask) -> None:
        """
        Process a single task.
        
        Args:
            task (PrioritizedTask): Task to process
        """
        result = None
        error = None
        
        try:
            # Check if the task is cancelled before execution
            if task.is_cancelled:
                logger.info(f"Task {task.task_id} ({task.name}) was cancelled before execution")
                return
            
            # Execute the task
            logger.info(f"Executing task {task.task_id} ({task.name}) with priority {task.priority}")
            start_time = time.time()
            result = task.execute_func(task.data)
            end_time = time.time()
            
            # Record completion
            with self.lock:
                self.completed_tasks[task.task_id] = {
                    "result": result,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time,
                    "priority": task.priority,
                    "deadline": task.deadline,
                    "name": task.name
                }
            
            logger.info(f"Task {task.task_id} ({task.name}) completed in {end_time - start_time:.2f} seconds")
            
        except Exception as e:
            error = str(e)
            logger.error(f"Error executing task {task.task_id} ({task.name}): {error}")
            
            # Record error
            with self.lock:
                self.completed_tasks[task.task_id] = {
                    "error": error,
                    "priority": task.priority,
                    "deadline": task.deadline,
                    "name": task.name
                }
        finally:
            # Clean up
            with self.lock:
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]
                self.current_workers -= 1
                self.condition.notify_all()  # Notify any waiting threads
    
    def schedule_task(self, 
                   name: str,
                   data: Dict[str, Any],
                   execute_func: Callable[[Dict[str, Any]], Any],
                   priority: Priority = Priority.MEDIUM,
                   deadline: Optional[float] = None,
                   task_id: Optional[str] = None) -> str:
        """
        Schedule a task for execution.
        
        Args:
            name (str): Human-readable task name
            data (Dict[str, Any]): Task data
            execute_func (Callable): Function to execute for this task
            priority (Priority): Task priority level
            deadline (Optional[float]): Task deadline timestamp
            task_id (Optional[str]): Unique task identifier (generated if None)
            
        Returns:
            str: Task identifier
        """
        # Generate a task ID if not provided
        if task_id is None:
            task_id = f"task_{int(time.time() * 1000)}_{id(execute_func)}"
        
        # Create the task
        task = PrioritizedTask(
            priority=priority.value,
            deadline=deadline or 0.0,
            create_time=time.time(),
            task_id=task_id,
            name=name,
            data=data,
            execute_func=execute_func
        )
        
        # Add the task to the queue
        with self.lock:
            heapq.heappush(self.task_queue, task)
            self.condition.notify_all()  # Notify worker threads
            
            logger.info(f"Task {task_id} ({name}) scheduled with priority {priority.name}")
        
        return task_id
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task if it hasn't started execution.
        
        Args:
            task_id (str): Task identifier
            
        Returns:
            bool: True if the task was found and cancelled, False otherwise
        """
        with self.lock:
            # Check if the task is in the active_tasks
            if task_id in self.active_tasks:
                self.active_tasks[task_id].is_cancelled = True
                logger.info(f"Task {task_id} marked as cancelled (may still complete if already executing)")
                return True
            
            # Look for the task in the queue
            for task in self.task_queue:
                if task.task_id == task_id:
                    task.is_cancelled = True
                    logger.info(f"Task {task_id} cancelled before execution")
                    return True
            
            # Task not found
            logger.warning(f"Task {task_id} not found for cancellation")
            return False
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of a task.
        
        Args:
            task_id (str): Task identifier
            
        Returns:
            Dict[str, Any]: Task status information
        """
        with self.lock:
            # Check if the task is completed
            if task_id in self.completed_tasks:
                result = self.completed_tasks[task_id].copy()
                result["status"] = "completed"
                return result
            
            # Check if the task is active
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                return {
                    "status": "running",
                    "priority": task.priority,
                    "deadline": task.deadline,
                    "name": task.name,
                    "create_time": task.create_time
                }
            
            # Look for the task in the queue
            for task in self.task_queue:
                if task.task_id == task_id:
                    return {
                        "status": "queued",
                        "priority": task.priority,
                        "deadline": task.deadline,
                        "name": task.name,
                        "create_time": task.create_time,
                        "is_cancelled": task.is_cancelled
                    }
            
            # Task not found
            return {"status": "unknown", "task_id": task_id}
    
    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get the status of the task queue.
        
        Returns:
            Dict[str, Any]: Queue status information
        """
        with self.lock:
            queue_tasks = len(self.task_queue)
            active_tasks = len(self.active_tasks)
            completed_tasks = len(self.completed_tasks)
            
            # Count tasks by priority
            priority_counts = {priority.name: 0 for priority in Priority}
            for task in self.task_queue:
                priority_name = Priority(task.priority).name
                priority_counts[priority_name] += 1
            
            # Find the next task to execute
            next_task = None
            if self.task_queue:
                # Create a copy of the top task without removing it
                next_task = self.task_queue[0]
                next_task_info = {
                    "task_id": next_task.task_id,
                    "name": next_task.name,
                    "priority": Priority(next_task.priority).name,
                    "deadline": datetime.fromtimestamp(next_task.deadline).isoformat() if next_task.deadline else None,
                    "create_time": datetime.fromtimestamp(next_task.create_time).isoformat()
                }
            else:
                next_task_info = None
            
            return {
                "queued_tasks": queue_tasks,
                "active_tasks": active_tasks,
                "completed_tasks": completed_tasks,
                "total_tasks": queue_tasks + active_tasks + completed_tasks,
                "priority_distribution": priority_counts,
                "next_task": next_task_info,
                "max_workers": self.max_workers,
                "current_workers": self.current_workers
            }
    
    def get_task_result(self, task_id: str, wait: bool = False, timeout: float = None) -> Optional[Dict[str, Any]]:
        """
        Get the result of a completed task.
        
        Args:
            task_id (str): Task identifier
            wait (bool): Whether to wait for task completion
            timeout (float): Maximum time to wait in seconds
            
        Returns:
            Optional[Dict[str, Any]]: Task result or None if not completed
        """
        if not wait:
            with self.lock:
                return self.completed_tasks.get(task_id)
        
        # Wait for the task to complete
        end_time = time.time() + timeout if timeout else None
        
        while True:
            with self.lock:
                # Check if the task is completed
                if task_id in self.completed_tasks:
                    return self.completed_tasks.get(task_id)
                
                # Check if the task exists
                task_exists = (task_id in self.active_tasks) or any(task.task_id == task_id for task in self.task_queue)
                if not task_exists:
                    logger.warning(f"Task {task_id} not found")
                    return None
                
                # Check timeout
                if end_time and time.time() >= end_time:
                    logger.warning(f"Timeout waiting for task {task_id}")
                    return None
                
                # Wait for a notification (with timeout)
                wait_time = end_time - time.time() if end_time else None
                self.condition.wait(wait_time)
    
    def adjust_task_priority(self, task_id: str, new_priority: Priority) -> bool:
        """
        Adjust the priority of a queued task.
        
        Args:
            task_id (str): Task identifier
            new_priority (Priority): New priority level
            
        Returns:
            bool: True if the task priority was adjusted, False otherwise
        """
        with self.lock:
            # Find the task in the queue
            task_index = None
            for i, task in enumerate(self.task_queue):
                if task.task_id == task_id:
                    task_index = i
                    break
            
            if task_index is not None:
                # Remove the task from the queue
                task = self.task_queue[task_index]
                self.task_queue[task_index] = self.task_queue[-1]
                self.task_queue.pop()
                
                # If the removed task was not at the end, sift it
                if task_index < len(self.task_queue):
                    heapq._siftup(self.task_queue, task_index)
                    heapq._siftdown(self.task_queue, 0, task_index)
                
                # Update the priority
                old_priority = task.priority
                task.priority = new_priority.value
                
                # Re-add the task to the queue
                heapq.heappush(self.task_queue, task)
                
                # Notify any waiting threads
                self.condition.notify_all()
                
                logger.info(f"Task {task_id} priority adjusted from {Priority(old_priority).name} to {new_priority.name}")
                return True
            
            # Check if the task is active
            if task_id in self.active_tasks:
                logger.warning(f"Cannot adjust priority of active task {task_id}")
                return False
            
            # Task not found
            logger.warning(f"Task {task_id} not found for priority adjustment")
            return False
    
    def update_task_deadline(self, task_id: str, new_deadline: float) -> bool:
        """
        Update the deadline of a queued task.
        
        Args:
            task_id (str): Task identifier
            new_deadline (float): New deadline timestamp
            
        Returns:
            bool: True if the task deadline was updated, False otherwise
        """
        with self.lock:
            # Find the task in the queue
            task_index = None
            for i, task in enumerate(self.task_queue):
                if task.task_id == task_id:
                    task_index = i
                    break
            
            if task_index is not None:
                # Remove the task from the queue
                task = self.task_queue[task_index]
                self.task_queue[task_index] = self.task_queue[-1]
                self.task_queue.pop()
                
                # If the removed task was not at the end, sift it
                if task_index < len(self.task_queue):
                    heapq._siftup(self.task_queue, task_index)
                    heapq._siftdown(self.task_queue, 0, task_index)
                
                # Update the deadline
                old_deadline = task.deadline
                task.deadline = new_deadline
                
                # Re-add the task to the queue
                heapq.heappush(self.task_queue, task)
                
                # Notify any waiting threads
                self.condition.notify_all()
                
                logger.info(f"Task {task_id} deadline updated from {datetime.fromtimestamp(old_deadline).isoformat()} to {datetime.fromtimestamp(new_deadline).isoformat()}")
                return True
            
            # Check if the task is active
            if task_id in self.active_tasks:
                logger.warning(f"Cannot update deadline of active task {task_id}")
                return False
            
            # Task not found
            logger.warning(f"Task {task_id} not found for deadline update")
            return False
    
    def clear_completed_tasks(self, age_seconds: Optional[float] = None) -> int:
        """
        Clear completed tasks from memory.
        
        Args:
            age_seconds (Optional[float]): Only clear tasks older than this age
            
        Returns:
            int: Number of tasks cleared
        """
        with self.lock:
            if age_seconds is None:
                # Clear all completed tasks
                count = len(self.completed_tasks)
                self.completed_tasks.clear()
                return count
            
            # Clear tasks older than the specified age
            current_time = time.time()
            min_completion_time = current_time - age_seconds
            
            tasks_to_clear = [
                task_id for task_id, result in self.completed_tasks.items()
                if "end_time" in result and result["end_time"] < min_completion_time
            ]
            
            for task_id in tasks_to_clear:
                del self.completed_tasks[task_id]
            
            return len(tasks_to_clear)
    
    def get_tasks_by_priority(self, priority: Priority) -> List[Dict[str, Any]]:
        """
        Get all queued tasks with the specified priority.
        
        Args:
            priority (Priority): Priority level to filter by
            
        Returns:
            List[Dict[str, Any]]: List of task information
        """
        with self.lock:
            tasks = []
            
            for task in self.task_queue:
                if task.priority == priority.value and not task.is_cancelled:
                    tasks.append({
                        "task_id": task.task_id,
                        "name": task.name,
                        "priority": Priority(task.priority).name,
                        "deadline": datetime.fromtimestamp(task.deadline).isoformat() if task.deadline else None,
                        "create_time": datetime.fromtimestamp(task.create_time).isoformat()
                    })
            
            return tasks
    
    def get_tasks_by_deadline(self, max_deadline: float) -> List[Dict[str, Any]]:
        """
        Get all tasks with a deadline before the specified time.
        
        Args:
            max_deadline (float): Maximum deadline timestamp
            
        Returns:
            List[Dict[str, Any]]: List of task information
        """
        with self.lock:
            tasks = []
            
            # Check queue
            for task in self.task_queue:
                if task.deadline <= max_deadline and not task.is_cancelled:
                    tasks.append({
                        "task_id": task.task_id,
                        "name": task.name,
                        "status": "queued",
                        "priority": Priority(task.priority).name,
                        "deadline": datetime.fromtimestamp(task.deadline).isoformat(),
                        "create_time": datetime.fromtimestamp(task.create_time).isoformat()
                    })
            
            # Check active tasks
            for task_id, task in self.active_tasks.items():
                if task.deadline <= max_deadline and not task.is_cancelled:
                    tasks.append({
                        "task_id": task.task_id,
                        "name": task.name,
                        "status": "running",
                        "priority": Priority(task.priority).name,
                        "deadline": datetime.fromtimestamp(task.deadline).isoformat(),
                        "create_time": datetime.fromtimestamp(task.create_time).isoformat()
                    })
            
            return tasks


class TimeSensitiveQuestionHandler:
    """
    Handler for time-sensitive questions and tasks.
    
    This class provides specialized handling for time-sensitive questions,
    prioritizing them appropriately and ensuring timely processing.
    
    Attributes:
        scheduler (PriorityScheduler): Underlying priority scheduler
        time_sensitivity_keywords (List[str]): Keywords indicating time sensitivity
        deadline_phrases (Dict[str, float]): Phrases indicating deadlines
    """
    
    def __init__(self, scheduler: Optional[PriorityScheduler] = None):
        """
        Initialize the time-sensitive question handler.
        
        Args:
            scheduler (Optional[PriorityScheduler]): Scheduler to use
        """
        self.scheduler = scheduler or PriorityScheduler()
        
        # Keywords that indicate time sensitivity
        self.time_sensitivity_keywords = [
            "urgent", "immediately", "asap", "emergency", "critical",
            "time-sensitive", "deadline", "today", "now", "right away",
            "quickly", "hurry", "rush", "紧急", "立即", "马上", "急需"
        ]
        
        # Phrases that indicate specific deadlines
        self.deadline_phrases = {
            "today": 1.0,           # 1 day
            "tomorrow": 2.0,        # 2 days
            "this week": 7.0,       # 7 days
            "next week": 14.0,      # 14 days
            "this month": 30.0,     # 30 days
            "within an hour": 0.042,  # 1 hour in days
            "within a day": 1.0,    # 1 day
        }
        
        # Start the scheduler
        self.scheduler.start()
        
        logger.info("TimeSensitiveQuestionHandler initialized")
    
    def detect_time_sensitivity(self, question: str) -> Tuple[Priority, Optional[float]]:
        """
        Detect time sensitivity from question text.
        
        Args:
            question (str): Question text
            
        Returns:
            Tuple[Priority, Optional[float]]: Detected priority and deadline
        """
        # Convert to lowercase for case-insensitive matching
        question_lower = question.lower()
        
        # Check for explicit time sensitivity keywords
        urgency_count = sum(1 for keyword in self.time_sensitivity_keywords 
                         if keyword in question_lower)
        
        # Determine priority based on urgency keywords
        if urgency_count >= 3:
            priority = Priority.CRITICAL
        elif urgency_count >= 2:
            priority = Priority.HIGH
        elif urgency_count >= 1:
            priority = Priority.MEDIUM
        else:
            priority = Priority.LOW
        
        # Check for deadline phrases
        deadline = None
        for phrase, days in self.deadline_phrases.items():
            if phrase in question_lower:
                # Convert days to timestamp
                deadline = time.time() + (days * 86400)
                break
        
        return priority, deadline
    
    def process_question(self, 
                       question: str, 
                       data: Dict[str, Any],
                       processor_func: Callable[[Dict[str, Any]], Any],
                       override_priority: Optional[Priority] = None,
                       override_deadline: Optional[float] = None) -> str:
        """
        Process a question with appropriate priority based on time sensitivity.
        
        Args:
            question (str): Question text
            data (Dict[str, Any]): Question data
            processor_func (Callable): Function to process the question
            override_priority (Optional[Priority]): Override the detected priority
            override_deadline (Optional[float]): Override the detected deadline
            
        Returns:
            str: Task identifier
        """
        # Detect time sensitivity if not overridden
        priority, deadline = self.detect_time_sensitivity(question)
        
        # Apply overrides if provided
        if override_priority is not None:
            priority = override_priority
        
        if override_deadline is not None:
            deadline = override_deadline
        
        # Prepare task data
        task_data = {
            "question": question,
            **data
        }
        
        # Schedule the task
        task_id = self.scheduler.schedule_task(
            name=f"Question: {question[:50]}...",
            data=task_data,
            execute_func=processor_func,
            priority=priority,
            deadline=deadline
        )
        
        logger.info(f"Question scheduled with priority {priority.name}, task_id={task_id}")
        return task_id
    
    def check_question_status(self, task_id: str) -> Dict[str, Any]:
        """
        Check the status of a scheduled question.
        
        Args:
            task_id (str): Task identifier
            
        Returns:
            Dict[str, Any]: Status information
        """
        return self.scheduler.get_task_status(task_id)
    
    def get_question_result(self, task_id: str, wait: bool = False, timeout: float = None) -> Optional[Dict[str, Any]]:
        """
        Get the result of a processed question.
        
        Args:
            task_id (str): Task identifier
            wait (bool): Whether to wait for completion
            timeout (float): Maximum time to wait in seconds
            
        Returns:
            Optional[Dict[str, Any]]: Question result or None if not completed
        """
        return self.scheduler.get_task_result(task_id, wait, timeout)
    
    def update_priority(self, task_id: str, new_priority: Priority) -> bool:
        """
        Update the priority of a scheduled question.
        
        Args:
            task_id (str): Task identifier
            new_priority (Priority): New priority level
            
        Returns:
            bool: True if the priority was updated, False otherwise
        """
        return self.scheduler.adjust_task_priority(task_id, new_priority)
    
    def set_urgent(self, task_id: str) -> bool:
        """
        Mark a question as urgent (CRITICAL priority).
        
        Args:
            task_id (str): Task identifier
            
        Returns:
            bool: True if the priority was updated, False otherwise
        """
        return self.scheduler.adjust_task_priority(task_id, Priority.CRITICAL)
    
    def get_urgent_questions(self) -> List[Dict[str, Any]]:
        """
        Get all queued urgent (CRITICAL priority) questions.
        
        Returns:
            List[Dict[str, Any]]: List of urgent question information
        """
        return self.scheduler.get_tasks_by_priority(Priority.CRITICAL)
    
    def get_overdue_questions(self) -> List[Dict[str, Any]]:
        """
        Get all questions with deadlines in the past.
        
        Returns:
            List[Dict[str, Any]]: List of overdue question information
        """
        return self.scheduler.get_tasks_by_deadline(time.time())
    
    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the handler.
        
        Args:
            wait (bool): Whether to wait for tasks to complete
        """
        self.scheduler.shutdown(wait)
        logger.info("TimeSensitiveQuestionHandler shut down") 