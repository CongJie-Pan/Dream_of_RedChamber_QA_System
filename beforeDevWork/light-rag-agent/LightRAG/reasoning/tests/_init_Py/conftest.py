"""
Pytest configuration for the reasoning module tests.

This module contains fixtures and configuration for pytest.
"""

import os
import sys
import pytest
import datetime
import traceback

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

# Ensure the test results base directory exists
if not os.path.exists(TEST_RESULTS_BASE_DIR):
    os.makedirs(TEST_RESULTS_BASE_DIR)

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
    # Extract the test module name from the nodeid (e.g., "test_init.py" from "test_init.py::TestInit::test_expected_imports")
    parts = nodeid.split("::")
    module_name = parts[0].replace(".py", "")
    
    # Create a folder for this test unit if it doesn't exist
    if module_name not in test_unit_folders:
        # Create a unique timestamped folder for this test run
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        test_unit_dir = os.path.join(TEST_RESULTS_BASE_DIR, f"{module_name}_{timestamp}")
        os.makedirs(test_unit_dir, exist_ok=True)
        test_unit_folders[module_name] = test_unit_dir
    
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
    
    return result_file

@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    """Set up the test environment."""
    # Nothing to do here as we'll create files on demand
    pass

@pytest.hookimpl(trylast=True)
def pytest_runtest_protocol(item, nextitem):
    """Custom test protocol to capture test execution and write results."""
    # Let pytest handle the test execution
    return None

@pytest.hookimpl(trylast=True)
def pytest_runtest_logreport(report):
    """Write test results to the document after each test."""
    if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
        # Get the appropriate file for this test
        result_file = get_test_result_file(report.nodeid)
        
        # Initialize the file if it doesn't exist yet
        if not os.path.exists(result_file):
            with open(result_file, 'w') as f:
                f.write(f"Test Results - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Test: {report.nodeid}\n")
                f.write("="*80 + "\n\n")
        
        # Append the test result
        with open(result_file, 'a') as f:
            f.write(f"Phase: {report.when}\n")
            f.write(f"Outcome: {report.outcome}\n")
            
            if report.outcome != "passed":
                f.write(f"Failure Information:\n")
                if hasattr(report, "longrepr"):
                    f.write(f"{report.longrepr}\n")
            
            f.write("-"*80 + "\n\n")

@pytest.fixture
def mock_deepseek_model(request):
    """
    Fixture that provides a mock DeepSeekModel class.
    
    This can be used in tests to avoid actual API calls to DeepSeek R1.
    """
    try:
        class MockDeepSeekModel:
            def __init__(self, api_key=None, model_name=None):
                self.api_key = api_key or "test_api_key"
                self.model_name = model_name or "deepseek-r1"
                
            def call(self, prompt, options=None):
                """Mock method for API calls"""
                return "This is a mock response from DeepSeek R1 model."
                
            def generate_chain_of_thought(self, query):
                """Mock method for CoT generation"""
                return [
                    "Step 1: Understand the question",
                    "Step 2: Analyze key components",
                    "Step 3: Formulate a response"
                ]
                
            def batch_call(self, prompts, options=None):
                """Mock method for batch API calls"""
                return ["Mock response" for _ in prompts]
        
        return MockDeepSeekModel
    
    except Exception as e:
        # Log the fixture error to the results file
        result_file = get_test_result_file(request.node.nodeid)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        
        with open(result_file, 'a') as f:
            f.write(f"ERROR in fixture mock_deepseek_model: {str(e)}\n")
            f.write(f"Traceback: {traceback.format_exc()}\n")
            f.write("-"*80 + "\n\n")
        raise  # Re-raise the exception to let pytest handle it

@pytest.fixture
def error_logger(request):
    """
    Fixture that provides a function to log errors to the test results file.
    
    Can be used in tests to log custom errors or additional information.
    """
    def _log_error(error_message, exception=None):
        # Get the appropriate file for this test
        result_file = get_test_result_file(request.node.nodeid)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        
        with open(result_file, 'a') as f:
            f.write(f"CUSTOM ERROR: {error_message}\n")
            if exception:
                f.write(f"Exception: {str(exception)}\n")
                f.write(f"Traceback: {traceback.format_exc()}\n")
            f.write("-"*80 + "\n\n")
    
    return _log_error 