# Adaptive Concurrency Module Tests

This directory contains tests for the `adaptive_concurrency.py` module, which provides adaptive concurrency control functionality for dynamically adjusting the number of concurrent tasks based on system load and resource availability.

## Test Files

- `conftest.py` - Contains pytest fixtures and configuration, including error logging functionality
- `test_adaptive_concurrency_manager.py` - Tests for the AdaptiveConcurrencyManager class
- `test_adaptive_batch_processor.py` - Tests for the AdaptiveBatchProcessor class
- `run_tests.py` - Helper script to run tests and ensure results are saved properly

## Running the Tests

### Using the Helper Script (Recommended)

The easiest way to run the tests is using the provided `run_tests.py` script, which ensures proper setup and verification of test results:

```bash
# From the reasoning directory
python -m tests.adaptive_concurrency_Py.run_tests

# Or from the adaptive_concurrency_Py directory
python run_tests.py
```

This script will:
1. Create a test run directory with timestamp
2. Run all tests with verbose output
3. Save all stdout and stderr to log files
4. Verify that test results were properly saved

### Using pytest Directly

Alternatively, you can run the tests using pytest directly:

```bash
# Run all tests in this directory
python -m pytest LightRAG/reasoning/tests/adaptive_concurrency_Py

# Run with verbose output
python -m pytest LightRAG/reasoning/tests/adaptive_concurrency_Py -v

# Run specific test file
python -m pytest LightRAG/reasoning/tests/adaptive_concurrency_Py/test_adaptive_concurrency_manager.py

# Run specific test class
python -m pytest LightRAG/reasoning/tests/adaptive_concurrency_Py/test_adaptive_concurrency_manager.py::TestAdaptiveConcurrencyManager

# Run specific test method
python -m pytest LightRAG/reasoning/tests/adaptive_concurrency_Py/test_adaptive_concurrency_manager.py::TestAdaptiveConcurrencyManager::test_init_with_default_values
```

## Test Results

Test results are automatically saved to the `LightRAG/reasoning/test_results` directory. Each test run creates multiple directories:

1. A timestamped directory for the test session (e.g., `test_run_20240625_120000/`)
2. Timestamped directories for each test module (e.g., `test_adaptive_concurrency_manager_20240625_120000/`)

Within these directories, individual test results are stored in separate files.

### Log Files

The following log files are generated:

- `session_[timestamp].log` - Overall test session log
- `summary_[timestamp].txt` - Summary of all tests run
- `[ClassName]_[test_method_name]_results.txt` - Individual test results
- `pytest.log` - When using the run script, contains pytest internal logs
- `stdout.log` and `stderr.log` - When using the run script, contains all console output

Example directory structure:
```
LightRAG/reasoning/test_results/
├── test_run_20240625_120000/
│   ├── pytest.log
│   ├── stdout.log
│   └── stderr.log
├── session_20240625_120000.log
├── summary_20240625_120000.txt
└── test_adaptive_concurrency_manager_20240625_120000/
    ├── TestAdaptiveConcurrencyManager_test_init_with_default_values_results.txt
    ├── TestAdaptiveConcurrencyManager_test_init_with_custom_values_results.txt
    └── ...
```

## Test Structure

Each test file includes:

1. Expected Use Cases - Tests for normal/typical usage scenarios
2. Edge Cases - Tests for boundary conditions or unusual inputs
3. Failure Cases - Tests for error handling and recovery mechanisms

## Error Handling

All tests include comprehensive error handling. Any errors during test execution are:
1. Logged to a file for later review
2. Include detailed information (error message, stack trace)
3. Stored in the test-specific results file

## Debugging Test Issues

If tests are not producing expected results or log files:

1. Run with the helper script which provides additional logging
2. Check console output for any error messages
3. Inspect the `stdout.log` and `stderr.log` files in the test run directory
4. Verify file permissions in the test_results directory

## Requirements

- pytest
- pytest-asyncio (for async tests)
- psutil 