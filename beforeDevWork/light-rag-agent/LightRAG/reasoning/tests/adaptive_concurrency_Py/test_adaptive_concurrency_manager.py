"""
Unit tests for the AdaptiveConcurrencyManager class in adaptive_concurrency.py.

This test module verifies the functionality of the AdaptiveConcurrencyManager class,
focusing on its ability to monitor system metrics and adaptively adjust concurrency.
"""

import unittest
import pytest
import time
import threading
from unittest.mock import patch, MagicMock
import os
import sys
import importlib

# Add parent directory to path if running standalone
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
grandparent_dir = os.path.dirname(parent_dir)
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)

# Import the module to test
from LightRAG.reasoning.adaptive_concurrency import AdaptiveConcurrencyManager, SystemMetrics, ApiMetrics


class TestAdaptiveConcurrencyManager(unittest.TestCase):
    """Tests for the AdaptiveConcurrencyManager class."""

    def setUp(self):
        """Set up the test case."""
        self.test_results_dir = os.path.join(parent_dir, "test_results")
        os.makedirs(self.test_results_dir, exist_ok=True)
        
        # Create a specific directory for this test module
        timestamp = importlib.import_module("datetime").datetime.now().strftime('%Y%m%d_%H%M%S')
        self.test_unit_dir = os.path.join(self.test_results_dir, f"test_adaptive_concurrency_manager_{timestamp}")
        os.makedirs(self.test_unit_dir, exist_ok=True)

        # Create a custom error logger for unittest (similar to pytest fixture)
        self.error_logger = self._create_error_logger()

    def _create_error_logger(self):
        """Create a local error logger function when not running under pytest."""
        def log_error(msg, exc=None):
            # Create a file for this test method
            method_name = self._testMethodName if hasattr(self, '_testMethodName') else 'unknown'
            log_file = os.path.join(self.test_unit_dir, f"TestAdaptiveConcurrencyManager_{method_name}_results.txt")
            
            import traceback
            with open(log_file, 'a') as f:
                f.write(f"ERROR: {msg}\n")
                if exc:
                    f.write(f"Exception: {str(exc)}\n")
                    f.write(f"Traceback: {traceback.format_exc()}\n")
                f.write("-"*80 + "\n\n")
        
        return log_error

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_io_counters')
    @patch('psutil.Process')
    @patch('psutil.pids')
    def test_init_with_default_values(self, mock_pids, mock_process, mock_disk_io, 
                                    mock_virtual_memory, mock_cpu_percent):
        """Test case for expected use - initialization with default values."""
        try:
            # Setup mocks
            mock_cpu_percent.return_value = 50.0
            mock_virtual_memory.return_value = MagicMock(percent=60.0)
            mock_disk_io.return_value = MagicMock(
                read_count=100, write_count=200, 
                read_bytes=1000, write_bytes=2000
            )
            mock_process.return_value.threads.return_value = [1, 2, 3]  # 3 threads
            mock_pids.return_value = [1, 2, 3, 4, 5]  # 5 processes
            
            # Create manager with default values
            manager = AdaptiveConcurrencyManager()
            
            # Verify initialization
            self.assertEqual(manager.max_concurrent_tasks, 5)  # Default from imported MAX_CONCURRENT_TASKS
            self.assertEqual(manager.current_concurrent_tasks, 2)  # Default: max_concurrent_tasks // 2
            self.assertEqual(manager.monitoring_interval, 5.0)
            self.assertFalse(manager.monitoring_active)
            self.assertIsNone(manager.monitor_thread)
            
            # Test initial metrics
            self.assertEqual(len(manager.system_metrics_history), 0)
            self.assertEqual(len(manager.api_metrics_history), 0)
            
        except Exception as e:
            self.error_logger("Error in test_init_with_default_values", e)
            raise

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_io_counters')
    @patch('psutil.Process')
    @patch('psutil.pids')
    def test_init_with_custom_values(self, mock_pids, mock_process, mock_disk_io, 
                                   mock_virtual_memory, mock_cpu_percent):
        """Test case for expected use - initialization with custom values."""
        try:
            # Setup mocks
            mock_cpu_percent.return_value = 50.0
            mock_virtual_memory.return_value = MagicMock(percent=60.0)
            mock_disk_io.return_value = MagicMock(
                read_count=100, write_count=200, 
                read_bytes=1000, write_bytes=2000
            )
            mock_process.return_value.threads.return_value = [1, 2, 3]  # 3 threads
            mock_pids.return_value = [1, 2, 3, 4, 5]  # 5 processes
            
            # Create manager with custom values
            custom_max = 10
            custom_initial = 4
            custom_interval = 2.5
            custom_history = 30
            
            manager = AdaptiveConcurrencyManager(
                initial_concurrent_tasks=custom_initial,
                max_concurrent_tasks=custom_max,
                monitoring_interval=custom_interval,
                metrics_history_size=custom_history
            )
            
            # Verify initialization
            self.assertEqual(manager.max_concurrent_tasks, custom_max)
            self.assertEqual(manager.current_concurrent_tasks, custom_initial)
            self.assertEqual(manager.monitoring_interval, custom_interval)
            self.assertEqual(manager.system_metrics_history.maxlen, custom_history)
            self.assertEqual(manager.api_metrics_history.maxlen, custom_history)
            
        except Exception as e:
            self.error_logger("Error in test_init_with_custom_values", e)
            raise

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_io_counters')
    @patch('psutil.Process')
    @patch('psutil.pids')
    def test_collect_system_metrics(self, mock_pids, mock_process, mock_disk_io, 
                                  mock_virtual_memory, mock_cpu_percent):
        """Test the collection of system metrics."""
        try:
            # Setup mocks
            mock_cpu_percent.return_value = 50.0
            mock_virtual_memory.return_value = MagicMock(percent=60.0)
            mock_disk_io.return_value = MagicMock(
                read_count=100, write_count=200, 
                read_bytes=1000, write_bytes=2000
            )
            mock_process.return_value.threads.return_value = [1, 2, 3]  # 3 threads
            mock_pids.return_value = [1, 2, 3, 4, 5]  # 5 processes
            
            # Create manager
            manager = AdaptiveConcurrencyManager()
            
            # Collect metrics
            metrics = manager._collect_system_metrics()
            
            # Verify metrics
            self.assertEqual(metrics.cpu_percent, 50.0)
            self.assertEqual(metrics.memory_percent, 60.0)
            self.assertEqual(metrics.thread_count, 3)
            self.assertEqual(metrics.process_count, 5)
            self.assertEqual(metrics.io_counters["read_count"], 100)
            self.assertEqual(metrics.io_counters["write_count"], 200)
            self.assertEqual(metrics.io_counters["read_bytes"], 1000)
            self.assertEqual(metrics.io_counters["write_bytes"], 2000)
            
        except Exception as e:
            self.error_logger("Error in test_collect_system_metrics", e)
            raise

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_io_counters')
    @patch('psutil.Process')
    @patch('psutil.pids')
    def test_adjust_concurrency_increase(self, mock_pids, mock_process, mock_disk_io, 
                                       mock_virtual_memory, mock_cpu_percent):
        """Test that concurrency increases when CPU usage is low."""
        try:
            # Setup mocks for low CPU usage
            mock_cpu_percent.return_value = 30.0
            mock_virtual_memory.return_value = MagicMock(percent=40.0)
            mock_disk_io.return_value = MagicMock(
                read_count=100, write_count=200, 
                read_bytes=1000, write_bytes=2000
            )
            mock_process.return_value.threads.return_value = [1, 2, 3]  # 3 threads
            mock_pids.return_value = [1, 2, 3, 4, 5]  # 5 processes
            
            # Create manager
            manager = AdaptiveConcurrencyManager(initial_concurrent_tasks=2, max_concurrent_tasks=10)
            
            # Add historical metrics to allow for increase
            for _ in range(3):
                metrics = SystemMetrics()
                metrics.cpu_percent = 30.0
                manager.system_metrics_history.append(metrics)
            
            # Get initial concurrency
            initial_concurrency = manager.current_concurrent_tasks
            
            # Collect and adjust based on metrics
            metrics = manager._collect_system_metrics()
            manager._adjust_concurrency(metrics)
            
            # Verify concurrency increased
            self.assertTrue(manager.current_concurrent_tasks > initial_concurrency)
            
        except Exception as e:
            self.error_logger("Error in test_adjust_concurrency_increase", e)
            raise

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_io_counters')
    @patch('psutil.Process')
    @patch('psutil.pids')
    def test_adjust_concurrency_decrease_high_cpu(self, mock_pids, mock_process, mock_disk_io, 
                                               mock_virtual_memory, mock_cpu_percent):
        """Test that concurrency decreases when CPU usage is high."""
        try:
            # Setup mocks for high CPU usage
            mock_cpu_percent.return_value = 85.0
            mock_virtual_memory.return_value = MagicMock(percent=60.0)
            mock_disk_io.return_value = MagicMock(
                read_count=100, write_count=200, 
                read_bytes=1000, write_bytes=2000
            )
            mock_process.return_value.threads.return_value = [1, 2, 3]  # 3 threads
            mock_pids.return_value = [1, 2, 3, 4, 5]  # 5 processes
            
            # Create manager
            manager = AdaptiveConcurrencyManager(initial_concurrent_tasks=5, max_concurrent_tasks=10)
            
            # Get initial concurrency
            initial_concurrency = manager.current_concurrent_tasks
            
            # Collect and adjust based on metrics
            metrics = manager._collect_system_metrics()
            manager._adjust_concurrency(metrics)
            
            # Verify concurrency decreased
            self.assertTrue(manager.current_concurrent_tasks < initial_concurrency)
            
        except Exception as e:
            self.error_logger("Error in test_adjust_concurrency_decrease_high_cpu", e)
            raise

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_io_counters')
    @patch('psutil.Process')
    @patch('psutil.pids')
    def test_adjust_concurrency_decrease_critical_cpu(self, mock_pids, mock_process, mock_disk_io, 
                                                   mock_virtual_memory, mock_cpu_percent):
        """Test that concurrency decreases significantly when CPU usage is critical."""
        try:
            # Setup mocks for critical CPU usage
            mock_cpu_percent.return_value = 95.0
            mock_virtual_memory.return_value = MagicMock(percent=60.0)
            mock_disk_io.return_value = MagicMock(
                read_count=100, write_count=200, 
                read_bytes=1000, write_bytes=2000
            )
            mock_process.return_value.threads.return_value = [1, 2, 3]  # 3 threads
            mock_pids.return_value = [1, 2, 3, 4, 5]  # 5 processes
            
            # Create manager
            manager = AdaptiveConcurrencyManager(initial_concurrent_tasks=8, max_concurrent_tasks=10)
            
            # Get initial concurrency
            initial_concurrency = manager.current_concurrent_tasks
            
            # Collect and adjust based on metrics
            metrics = manager._collect_system_metrics()
            manager._adjust_concurrency(metrics)
            
            # Verify concurrency decreased significantly (by half for critical load)
            self.assertEqual(manager.current_concurrent_tasks, initial_concurrency // 2)
            
        except Exception as e:
            self.error_logger("Error in test_adjust_concurrency_decrease_critical_cpu", e)
            raise

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_io_counters')
    @patch('psutil.Process')
    @patch('psutil.pids')
    def test_edge_case_minimum_concurrency(self, mock_pids, mock_process, mock_disk_io, 
                                         mock_virtual_memory, mock_cpu_percent):
        """Test edge case - concurrency doesn't go below 1."""
        try:
            # Setup mocks for critical CPU usage
            mock_cpu_percent.return_value = 95.0
            mock_virtual_memory.return_value = MagicMock(percent=60.0)
            mock_disk_io.return_value = MagicMock(
                read_count=100, write_count=200, 
                read_bytes=1000, write_bytes=2000
            )
            mock_process.return_value.threads.return_value = [1, 2, 3]  # 3 threads
            mock_pids.return_value = [1, 2, 3, 4, 5]  # 5 processes
            
            # Create manager with initial concurrency of 1
            manager = AdaptiveConcurrencyManager(initial_concurrent_tasks=1, max_concurrent_tasks=10)
            
            # Collect and adjust based on metrics
            metrics = manager._collect_system_metrics()
            manager._adjust_concurrency(metrics)
            
            # Verify concurrency remains at 1
            self.assertEqual(manager.current_concurrent_tasks, 1)
            
        except Exception as e:
            self.error_logger("Error in test_edge_case_minimum_concurrency", e)
            raise

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_io_counters')
    @patch('psutil.Process')
    @patch('psutil.pids')
    def test_edge_case_maximum_concurrency(self, mock_pids, mock_process, mock_disk_io, 
                                         mock_virtual_memory, mock_cpu_percent):
        """Test edge case - concurrency doesn't exceed maximum."""
        try:
            # Setup mocks for low CPU usage
            mock_cpu_percent.return_value = 30.0
            mock_virtual_memory.return_value = MagicMock(percent=40.0)
            mock_disk_io.return_value = MagicMock(
                read_count=100, write_count=200, 
                read_bytes=1000, write_bytes=2000
            )
            mock_process.return_value.threads.return_value = [1, 2, 3]  # 3 threads
            mock_pids.return_value = [1, 2, 3, 4, 5]  # 5 processes
            
            # Create manager with initial concurrency at max
            max_concurrency = 10
            manager = AdaptiveConcurrencyManager(
                initial_concurrent_tasks=max_concurrency, 
                max_concurrent_tasks=max_concurrency
            )
            
            # Add historical metrics to allow for increase attempt
            for _ in range(3):
                metrics = SystemMetrics()
                metrics.cpu_percent = 30.0
                manager.system_metrics_history.append(metrics)
            
            # Collect and adjust based on metrics
            metrics = manager._collect_system_metrics()
            manager._adjust_concurrency(metrics)
            
            # Verify concurrency doesn't exceed maximum
            self.assertEqual(manager.current_concurrent_tasks, max_concurrency)
            
        except Exception as e:
            self.error_logger("Error in test_edge_case_maximum_concurrency", e)
            raise

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_io_counters')
    @patch('psutil.Process')
    @patch('psutil.pids')
    def test_failure_case_collect_metrics_exception(self, mock_pids, mock_process, mock_disk_io, 
                                                 mock_virtual_memory, mock_cpu_percent):
        """Test failure case - exception during metrics collection."""
        try:
            # Setup mocks to raise exception
            mock_cpu_percent.side_effect = Exception("Simulated CPU metric error")
            
            # Create manager
            manager = AdaptiveConcurrencyManager()
            
            # Collect metrics - should handle exception gracefully
            metrics = manager._collect_system_metrics()
            
            # Verify default metrics are returned
            self.assertEqual(metrics.cpu_percent, 0.0)
            self.assertEqual(metrics.memory_percent, 0.0)
            self.assertEqual(metrics.thread_count, 0)
            self.assertEqual(metrics.process_count, 0)
            
        except Exception as e:
            self.error_logger("Error in test_failure_case_collect_metrics_exception", e)
            raise

    def test_failure_case_monitoring_exception(self):
        """Test failure case - exception during monitoring loop."""
        try:
            # Create a manager that will raise exception during monitoring
            manager = AdaptiveConcurrencyManager()
            
            # Patch the _collect_system_metrics method to raise exception
            original_collect = manager._collect_system_metrics
            
            def raise_exception(*args, **kwargs):
                raise Exception("Simulated monitoring error")
                
            manager._collect_system_metrics = raise_exception
            
            # Start monitoring
            manager.start_monitoring()
            
            # Let monitoring run for a short time
            time.sleep(0.5)
            
            # Stop monitoring
            manager.stop_monitoring()
            
            # Reset method
            manager._collect_system_metrics = original_collect
            
            # No assertion needed - test passes if no uncaught exception is raised
            
        except Exception as e:
            self.error_logger("Error in test_failure_case_monitoring_exception", e)
            raise


# Run pytest compatible tests
@pytest.mark.usefixtures("mock_psutil")
def test_adaptive_concurrency_manager_init(mock_psutil, error_logger):
    """Test initialization of AdaptiveConcurrencyManager with pytest fixtures."""
    try:
        # Set mock CPU usage
        mock_psutil.set_cpu_percent(50.0)
        mock_psutil.set_memory_percent(60.0)
        
        # Create manager with default values
        manager = AdaptiveConcurrencyManager()
        
        # Verify initialization
        assert manager.max_concurrent_tasks == 5  # Default from imported MAX_CONCURRENT_TASKS
        assert manager.current_concurrent_tasks == 2  # Default: max_concurrent_tasks // 2
        assert manager.monitoring_interval == 5.0
        assert not manager.monitoring_active
        assert manager.monitor_thread is None
        
        # Test initial metrics
        assert len(manager.system_metrics_history) == 0
        assert len(manager.api_metrics_history) == 0
        
    except Exception as e:
        error_logger("Error in test_adaptive_concurrency_manager_init", e)
        raise

@pytest.mark.usefixtures("mock_psutil")
def test_adaptive_concurrency_manager_record_api_call(mock_psutil, error_logger):
    """Test recording API calls and metrics tracking."""
    try:
        manager = AdaptiveConcurrencyManager()
        
        # Record successful API call
        manager.record_api_call(
            success=True,
            latency=0.2,
            rate_limited=False,
            token_count=100
        )
        
        # Verify metrics updated
        assert manager.current_api_metrics.total_calls == 1
        assert manager.current_api_metrics.successful_calls == 1
        assert manager.current_api_metrics.failed_calls == 0
        assert manager.current_api_metrics.rate_limit_hits == 0
        assert manager.current_api_metrics.average_latency == 0.2
        assert manager.current_api_metrics.token_usage == 100
        
        # Record failed API call with rate limit
        manager.record_api_call(
            success=False,
            latency=0.3,
            rate_limited=True,
            token_count=50
        )
        
        # Verify metrics updated
        assert manager.current_api_metrics.total_calls == 2
        assert manager.current_api_metrics.successful_calls == 1
        assert manager.current_api_metrics.failed_calls == 1
        assert manager.current_api_metrics.rate_limit_hits == 1
        assert abs(manager.current_api_metrics.average_latency - 0.25) < 0.001  # ~0.25 (average of 0.2 and 0.3)
        assert manager.current_api_metrics.token_usage == 150
        
    except Exception as e:
        error_logger("Error in test_adaptive_concurrency_manager_record_api_call", e)
        raise


if __name__ == '__main__':
    unittest.main() 