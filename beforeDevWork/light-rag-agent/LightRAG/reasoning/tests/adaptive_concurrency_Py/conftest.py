"""
Pytest configuration for the adaptive_concurrency module tests.

This module contains fixtures and configuration for pytest.
"""

import os
import sys
import pytest
import datetime
import traceback
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("adaptive_concurrency_tests")

# Add the parent directory to the path so we can import the module
# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (reasoning module)
parent_dir = os.path.dirname(script_dir)
# Get the grandparent directory (LightRAG)
grandparent_dir = os.path.dirname(parent_dir)
# Add the grandparent directory to sys.path
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)

# Define the base path for test results
TEST_RESULTS_BASE_DIR = os.path.join(parent_dir, "test_results")
logger.info(f"Test results will be stored in: {TEST_RESULTS_BASE_DIR}")

# Ensure the test results base directory exists
if not os.path.exists(TEST_RESULTS_BASE_DIR):
    os.makedirs(TEST_RESULTS_BASE_DIR)
    logger.info(f"Created test results directory: {TEST_RESULTS_BASE_DIR}")

# Dictionary to store test module/test unit name to folder mapping
test_unit_folders = {}

def get_test_result_file(nodeid):
    """
    Get the appropriate file path for a test result based on test unit name.
    
    Args:
        nodeid: The pytest node ID of the test
        
    Returns:
        str: The full path to the test results file
    """
    # Extract the test module name from the nodeid (e.g., "test_adaptive_concurrency.py" from "test_adaptive_concurrency.py::TestAdaptiveConcurrencyManager::test_init")
    parts = nodeid.split("::")
    module_name = parts[0].replace(".py", "")
    
    # Create a folder for this test unit if it doesn't exist
    if module_name not in test_unit_folders:
        # Create a unique timestamped folder for this test run
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        test_unit_dir = os.path.join(TEST_RESULTS_BASE_DIR, f"{module_name}_{timestamp}")
        os.makedirs(test_unit_dir, exist_ok=True)
        test_unit_folders[module_name] = test_unit_dir
        logger.info(f"Created test unit directory: {test_unit_dir}")
    
    # Get the folder for this test unit
    test_unit_dir = test_unit_folders[module_name]
    
    # Create the full file path for the test results
    if len(parts) > 1:
        # If there's a class and method, include them in the filename
        if len(parts) > 2:
            class_name = parts[1]
            method_name = parts[2]
            result_file = os.path.join(test_unit_dir, f"{class_name}_{method_name}_results.txt")
        else:
            # Just a function test
            func_name = parts[1]
            result_file = os.path.join(test_unit_dir, f"{func_name}_results.txt")
    else:
        # Fallback to just using the module name
        result_file = os.path.join(test_unit_dir, "results.txt")
    
    logger.info(f"Test result file for {nodeid}: {result_file}")
    return result_file

@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    """Set up the test environment."""
    logger.info("pytest_configure hook called")
    # Create a timestamp for this test session
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create a session log file
    session_log_file = os.path.join(TEST_RESULTS_BASE_DIR, f"session_{timestamp}.log")
    with open(session_log_file, 'w') as f:
        f.write(f"Test Session Started - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
    
    logger.info(f"Session log file created: {session_log_file}")

@pytest.hookimpl(trylast=True)
def pytest_runtest_protocol(item, nextitem):
    """Custom test protocol to capture test execution and write results."""
    logger.info(f"Running test: {item.nodeid}")
    # Let pytest handle the test execution
    return None

@pytest.hookimpl(trylast=True)
def pytest_runtest_logreport(report):
    """Write test results to the document after each test."""
    logger.info(f"Test report for {report.nodeid}: {report.when} - {report.outcome}")
    
    try:
        if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
            # Get the appropriate file for this test
            result_file = get_test_result_file(report.nodeid)
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(result_file), exist_ok=True)
            
            # Initialize the file if it doesn't exist yet
            if not os.path.exists(result_file):
                with open(result_file, 'w') as f:
                    f.write(f"Test Results - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Test: {report.nodeid}\n")
                    f.write("="*80 + "\n\n")
                logger.info(f"Created result file: {result_file}")
            
            # Append the test result
            with open(result_file, 'a') as f:
                f.write(f"Phase: {report.when}\n")
                f.write(f"Outcome: {report.outcome}\n")
                
                if report.outcome != "passed":
                    f.write(f"Failure Information:\n")
                    if hasattr(report, "longrepr"):
                        f.write(f"{report.longrepr}\n")
                
                f.write("-"*80 + "\n\n")
            logger.info(f"Appended result to file: {result_file}")
    except Exception as e:
        logger.error(f"Error in pytest_runtest_logreport: {e}")
        logger.error(traceback.format_exc())

@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    """Called after whole test run finished."""
    logger.info(f"Test session finished with exit status: {exitstatus}")
    
    # Log the summary
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = os.path.join(TEST_RESULTS_BASE_DIR, f"summary_{timestamp}.txt")
    
    try:
        with open(summary_file, 'w') as f:
            f.write(f"Test Session Summary - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Exit Status: {exitstatus}\n")
            f.write(f"Test Directories Created:\n")
            for module, directory in test_unit_folders.items():
                f.write(f"  - {module}: {directory}\n")
            f.write("="*80 + "\n\n")
        logger.info(f"Test summary written to: {summary_file}")
    except Exception as e:
        logger.error(f"Error writing summary file: {e}")
        logger.error(traceback.format_exc())

@pytest.fixture
def error_logger(request):
    """
    Fixture that provides a function to log errors to the test results file.
    
    Can be used in tests to log custom errors or additional information.
    """
    def _log_error(error_message, exception=None):
        try:
            # Get the appropriate file for this test
            result_file = get_test_result_file(request.node.nodeid)
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(result_file), exist_ok=True)
            
            # Log to console as well
            logger.error(f"ERROR in {request.node.nodeid}: {error_message}")
            if exception:
                logger.error(f"Exception: {str(exception)}")
            
            with open(result_file, 'a') as f:
                f.write(f"CUSTOM ERROR: {error_message}\n")
                if exception:
                    f.write(f"Exception: {str(exception)}\n")
                    f.write(f"Traceback: {traceback.format_exc()}\n")
                f.write("-"*80 + "\n\n")
            
            logger.info(f"Error logged to: {result_file}")
        except Exception as e:
            logger.error(f"Error in error_logger fixture: {e}")
            logger.error(traceback.format_exc())
    
    return _log_error

@pytest.fixture
def mock_psutil(monkeypatch):
    """
    Fixture that provides a mock psutil module for testing.
    """
    class MockProcess:
        def threads(self):
            return [1, 2, 3]  # Simulate 3 threads
    
    class MockCpuPercent:
        def __init__(self, return_value=50.0):
            self.return_value = return_value
        
        def __call__(self, interval=None):
            return self.return_value
    
    class MockVirtualMemory:
        def __init__(self, percent=50.0):
            self.percent = percent
    
    class MockIOCounters:
        def __init__(self):
            self.read_count = 100
            self.write_count = 200
            self.read_bytes = 1000
            self.write_bytes = 2000
    
    class MockPsUtil:
        def __init__(self):
            self.cpu_percent_mock = MockCpuPercent()
            self.virtual_memory_mock = MockVirtualMemory()
            self.io_counters_mock = MockIOCounters()
            self.process_mock = MockProcess()
            self.pids_list = [1, 2, 3, 4, 5]  # 5 processes
        
        def cpu_percent(self, interval=None):
            return self.cpu_percent_mock(interval)
        
        def virtual_memory(self):
            return self.virtual_memory_mock
        
        def disk_io_counters(self):
            return self.io_counters_mock
        
        def Process(self, pid=None):
            return self.process_mock
        
        def pids(self):
            return self.pids_list
        
        # Methods to control mocked behavior
        def set_cpu_percent(self, value):
            self.cpu_percent_mock.return_value = value
        
        def set_memory_percent(self, value):
            self.virtual_memory_mock.percent = value
    
    # Create the mock and patch psutil
    mock_psutil_instance = MockPsUtil()
    monkeypatch.setattr("psutil.cpu_percent", mock_psutil_instance.cpu_percent)
    monkeypatch.setattr("psutil.virtual_memory", mock_psutil_instance.virtual_memory)
    monkeypatch.setattr("psutil.disk_io_counters", mock_psutil_instance.disk_io_counters)
    monkeypatch.setattr("psutil.Process", mock_psutil_instance.Process)
    monkeypatch.setattr("psutil.pids", mock_psutil_instance.pids)
    
    logger.info("Mock psutil fixture initialized")
    return mock_psutil_instance 