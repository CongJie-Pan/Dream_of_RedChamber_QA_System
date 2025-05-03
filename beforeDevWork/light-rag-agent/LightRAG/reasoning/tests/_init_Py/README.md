# Reasoning Module Tests

This directory contains unit tests for the LightRAG reasoning module.

## Running Tests

To run all tests:

To run a specific test file:

```bash
python -m pytest LightRAG/reasoning/tests/_init_Py/test_init.py
```

To run with verbose output:

```bash
python -m pytest LightRAG/reasoning/_init_Py/tests -v
```

## Test Structure

- `__init__.py` - Makes the tests directory a Python package
- `conftest.py` - Contains pytest fixtures and configuration
- `test_init.py` - Tests for the reasoning module's `__init__.py`
- Additional test files will be added for each module component

## Test Coverage

These tests ensure:
1. The module properly exports all required classes
2. The expected use cases function correctly
3. Edge cases are handled appropriately
4. Failure scenarios are handled gracefully

## Writing New Tests

When adding new components to the reasoning module, please add corresponding test files following these guidelines:

1. Create a test file named `test_<module_name>.py`
2. Include at least:
   - One test for expected use
   - One test for an edge case
   - One test for a failure case
3. Use mocks where appropriate to avoid external dependencies
4. Follow the pytest best practices 