# Agent.py Tests

This directory contains tests for the agent.py module in the LightRAG reasoning package.

## Overview

The tests in this directory focus on the following classes from the agent.py module:

1. `ReasoningStepLogger` - Responsible for tracking and visualizing reasoning steps
2. `ReasoningAgent` - Core reasoning agent for complex query analysis and decomposition

## Test Structure

The tests follow a comprehensive approach with:

- Unit tests using the Python `unittest` framework
- Integration with pytest for fixtures and additional test functionality
- Dedicated error handling and logging mechanisms
- Organized output storage in test-specific directories

## Test Categories

For each class, we've implemented tests for:

1. **Normal Cases** - Testing expected behavior with valid inputs
2. **Edge Cases** - Testing boundary conditions and unusual inputs
3. **Failure Cases** - Testing error handling and graceful degradation

## Running the Tests

You can run the tests using pytest:

```bash
cd LightRAG/reasoning
python -m pytest tests/agent_Py -v
```

Or using unittest:

```bash
cd LightRAG/reasoning
python -m unittest discover -s tests/agent_Py
```

## Test Results

Test results are automatically saved to the `tests/test_results` directory with the following organization:

- Each test module has its own timestamped directory
- Each test method has its own results file
- Errors and exceptions are logged with full stack traces
- Summary files provide an overview of the test session

## Test Fixtures

The tests use several fixtures:

- `mock_model` - Provides a mock DeepSeekModel for testing
- `mock_cot` - Provides a mock ChainOfThought for testing
- `error_logger` - Provides a function to log errors to the test results file

## Error Handling

All tests include comprehensive error handling with:

- Detailed error messages
- Full exception stack traces
- Logging to console and files
- Test result files containing error information

## Output Storage Organization

Test outputs are organized as follows:

```
LightRAG/reasoning/tests/test_results/
├── test_agent_YYYYMMDD_HHMMSS/
│   ├── TestReasoningStepLogger_test_start_session_normal_results.txt
│   ├── TestReasoningStepLogger_test_log_step_with_large_data_results.txt
│   ├── ...
│   ├── TestReasoningAgent_test_analyze_query_normal_results.txt
│   ├── ...
├── session_YYYYMMDD_HHMMSS.log
└── summary_YYYYMMDD_HHMMSS.txt
```

Each test run creates a unique timestamped directory to prevent overwriting previous results. 