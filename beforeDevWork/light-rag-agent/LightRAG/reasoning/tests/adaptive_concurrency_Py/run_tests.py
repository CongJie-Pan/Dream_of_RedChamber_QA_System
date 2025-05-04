#!/usr/bin/env python
"""
Script to run the adaptive_concurrency module tests and ensure results are properly saved.

This script will:
1. Run all the tests in the adaptive_concurrency_Py directory
2. Ensure the test results directory exists
3. Verify that test results are being correctly saved
"""

import os
import sys
import subprocess
import datetime
import shutil

# Get the current directory (where this script is located)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (reasoning module)
parent_dir = os.path.dirname(script_dir)
# Get the grandparent directory (LightRAG)
grandparent_dir = os.path.dirname(parent_dir)

# Define the test results directory
TEST_RESULTS_DIR = os.path.join(parent_dir, "test_results")

def setup_test_environment():
    """Set up the test environment before running tests."""
    print(f"Setting up test environment...")
    
    # Ensure the test results directory exists
    if not os.path.exists(TEST_RESULTS_DIR):
        os.makedirs(TEST_RESULTS_DIR)
        print(f"Created test results directory: {TEST_RESULTS_DIR}")
    
    # Create a timestamp for this test run
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create a run-specific directory for this test execution
    run_dir = os.path.join(TEST_RESULTS_DIR, f"test_run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Created run directory: {run_dir}")
    
    # Return the run directory
    return run_dir

def run_tests(run_dir, verbose=True):
    """Run the tests and ensure results are saved."""
    print(f"Running tests...")
    
    # Build the pytest command
    cmd = [
        sys.executable, 
        "-m", 
        "pytest",
        os.path.basename(script_dir),  # Run tests in this directory
        "-v" if verbose else "",       # Verbose output if requested
        "--capture=tee-sys",           # Capture output while still showing it
        f"--log-file={os.path.join(run_dir, 'pytest.log')}"  # Save pytest logs
    ]
    cmd = [c for c in cmd if c]  # Remove empty strings
    
    # Print the command being run
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the command from the parent directory
    original_dir = os.getcwd()
    os.chdir(parent_dir)
    
    try:
        # Run pytest and capture the output
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Save the output to the run directory
        with open(os.path.join(run_dir, "stdout.log"), "w") as f:
            f.write(result.stdout)
        
        with open(os.path.join(run_dir, "stderr.log"), "w") as f:
            f.write(result.stderr)
        
        print(f"Tests completed with exit code: {result.returncode}")
        return result.returncode
    
    finally:
        # Return to the original directory
        os.chdir(original_dir)

def verify_test_results():
    """Verify that test results were saved correctly."""
    print(f"Verifying test results...")
    
    # Check if the test results directory exists
    if not os.path.exists(TEST_RESULTS_DIR):
        print(f"ERROR: Test results directory does not exist: {TEST_RESULTS_DIR}")
        return False
    
    # Get all subdirectories in the test results directory
    subdirs = [d for d in os.listdir(TEST_RESULTS_DIR) 
              if os.path.isdir(os.path.join(TEST_RESULTS_DIR, d))]
    
    if not subdirs:
        print(f"ERROR: No test result subdirectories found in: {TEST_RESULTS_DIR}")
        return False
    
    # Check the most recent directory for test result files
    latest_subdir = max(subdirs, key=lambda d: os.path.getmtime(os.path.join(TEST_RESULTS_DIR, d)))
    latest_dir_path = os.path.join(TEST_RESULTS_DIR, latest_subdir)
    
    print(f"Most recent test directory: {latest_dir_path}")
    
    # List all files in the latest directory
    files = [f for f in os.listdir(latest_dir_path) 
            if os.path.isfile(os.path.join(latest_dir_path, f))]
    
    if not files:
        print(f"ERROR: No test result files found in: {latest_dir_path}")
        return False
    
    print(f"Found {len(files)} test result files in: {latest_dir_path}")
    
    # Print the first few files
    for i, file in enumerate(files[:5]):
        print(f"  - {file}")
    
    if len(files) > 5:
        print(f"  - ... and {len(files) - 5} more files")
    
    return True

def main():
    """Main function to run the tests."""
    print("=" * 80)
    print(f"Running adaptive_concurrency module tests")
    print(f"Date and Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Setup test environment
    run_dir = setup_test_environment()
    
    # Run the tests
    exit_code = run_tests(run_dir, verbose=True)
    
    # Verify test results
    success = verify_test_results()
    
    if success:
        print("\nTest results were successfully saved.")
    else:
        print("\nWARNING: Test results may not have been saved correctly.")
    
    print("\nTest run complete.\n")
    
    # Return the exit code
    return exit_code

if __name__ == "__main__":
    sys.exit(main()) 