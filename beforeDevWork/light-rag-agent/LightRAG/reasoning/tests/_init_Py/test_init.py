"""
Unit tests for the reasoning module's __init__.py.

This test module verifies that the reasoning module's __init__.py correctly exports
the expected classes and that these classes can be imported directly from the package.
"""

import unittest
import sys
import importlib
import traceback
import os
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the module
import os
import sys

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (reasoning module)
parent_dir = os.path.dirname(script_dir)
# Get the grandparent directory (LightRAG)
grandparent_dir = os.path.dirname(parent_dir)
# Add the grandparent directory to sys.path
sys.path.insert(0, grandparent_dir)


class TestInit(unittest.TestCase):
    """Tests for the reasoning module's __init__.py"""

    def setUp(self):
        """Set up the test case, including error logging mechanism."""
        # Setup the test results directory
        test_results_dir = os.path.join(parent_dir, "test_results")
        os.makedirs(test_results_dir, exist_ok=True)
        
        # Create a specific directory for this test module
        timestamp = importlib.import_module("datetime").datetime.now().strftime('%Y%m%d_%H%M%S')
        self.test_unit_dir = os.path.join(test_results_dir, f"test_init_{timestamp}")
        os.makedirs(self.test_unit_dir, exist_ok=True)
        
        # Import the error_logger fixture from conftest.py if we're running in pytest
        try:
            import pytest
            if hasattr(pytest, "error_logger"):
                self.error_logger = pytest.error_logger
            else:
                # If not running in pytest or error_logger is not available
                self.error_logger = self._create_local_error_logger()
        except ImportError:
            # If pytest is not available, create a simple error logger
            self.error_logger = self._create_local_error_logger()
    
    def _create_local_error_logger(self):
        """Create a local error logger function when not running under pytest."""
        def log_error(msg, exc=None):
            # Create a file for this test method
            method_name = self._testMethodName if hasattr(self, '_testMethodName') else 'unknown'
            log_file = os.path.join(self.test_unit_dir, f"TestInit_{method_name}_results.txt")
            
            with open(log_file, 'a') as f:
                f.write(f"ERROR: {msg}\n")
                if exc:
                    f.write(f"Exception: {str(exc)}\n")
                    f.write(f"Traceback: {traceback.format_exc()}\n")
                f.write("-"*80 + "\n\n")
        
        return log_error

    def test_expected_imports(self):
        """Test case for expected use - all classes can be imported properly."""
        try:
            # Import the module
            from LightRAG.reasoning import DeepSeekModel, ChainOfThought, ReasoningAgent, ReasoningPipeline
            
            # Verify that each class is defined
            self.assertIsNotNone(DeepSeekModel)
            self.assertIsNotNone(ChainOfThought)
            self.assertIsNotNone(ReasoningAgent)
            self.assertIsNotNone(ReasoningPipeline)
            
            # Verify class types (they should be classes, not instances)
            self.assertTrue(isinstance(DeepSeekModel, type))
            self.assertTrue(isinstance(ChainOfThought, type))
            self.assertTrue(isinstance(ReasoningAgent, type))
            self.assertTrue(isinstance(ReasoningPipeline, type))
        except Exception as e:
            self.error_logger("Error in test_expected_imports", e)
            raise  # Re-raise to have the test fail
    
    def test_all_variable(self):
        """Test edge case - verify the __all__ variable contains the expected classes."""
        try:
            # Import the module
            import LightRAG.reasoning as reasoning
            
            # Check that __all__ contains exactly the expected classes
            expected_all = ['DeepSeekModel', 'ChainOfThought', 'ReasoningAgent', 'ReasoningPipeline']
            self.assertEqual(set(reasoning.__all__), set(expected_all))
            
            # Check that each element in __all__ is a class that can be accessed from the module
            for class_name in reasoning.__all__:
                # This will raise an AttributeError if the class doesn't exist
                cls = getattr(reasoning, class_name)
                self.assertTrue(isinstance(cls, type))
        except Exception as e:
            self.error_logger("Error in test_all_variable", e)
            raise  # Re-raise to have the test fail
    
    @patch.dict('sys.modules')
    def test_import_failure(self):
        """Test failure case - when an imported module raises an ImportError."""
        try:
            # Simulate an ImportError when importing models
            sys.modules['LightRAG.reasoning.models'] = MagicMock(side_effect=ImportError("Mocked ImportError"))
            
            # Reload the module to simulate the import error
            with self.assertRaises(ImportError):
                # This should raise an ImportError because models.py can't be imported
                import importlib
                importlib.reload(sys.modules.get('LightRAG.reasoning', None))
                import LightRAG.reasoning
        except Exception as e:
            if not isinstance(e, ImportError):
                # If it's another kind of error (not the expected ImportError)
                self.error_logger("Unexpected error in test_import_failure", e)
            raise  # Re-raise to have the test fail or pass as appropriate


# Add a pytest compatible test function to use the fixtures directly
def test_with_pytest_fixtures(mock_deepseek_model, error_logger):
    """Test using pytest fixtures to demonstrate the error logging."""
    try:
        # Create an instance of our mock model
        model = mock_deepseek_model()
        
        # Test basic functionality
        response = model.call("Test prompt")
        assert "mock response" in response.lower(), "Mock model should return a mock response"
        
        # Test chain of thought generation
        cot_steps = model.generate_chain_of_thought("Test query")
        assert len(cot_steps) == 3, "Should return 3 steps for chain of thought"
        assert "Step 1" in cot_steps[0], "First step should be labeled as Step 1"
        
        # Test batch call functionality
        batch_responses = model.batch_call(["Query 1", "Query 2"])
        assert len(batch_responses) == 2, "Should return responses for both prompts"
    except Exception as e:
        error_logger("Error in test_with_pytest_fixtures", e)
        raise  # Re-raise to have the test fail


if __name__ == '__main__':
    unittest.main() 