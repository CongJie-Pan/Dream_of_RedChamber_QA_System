"""
Unit tests for the reasoning module's __init__.py.

This test module verifies that the reasoning module's __init__.py correctly exports
the expected classes and that these classes can be imported directly from the package.
"""

import unittest
import sys
import importlib
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

    def test_expected_imports(self):
        """Test case for expected use - all classes can be imported properly."""
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
    
    def test_all_variable(self):
        """Test edge case - verify the __all__ variable contains the expected classes."""
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
    
    @patch.dict('sys.modules')
    def test_import_failure(self):
        """Test failure case - when an imported module raises an ImportError."""
        # Simulate an ImportError when importing models
        sys.modules['LightRAG.reasoning.models'] = MagicMock(side_effect=ImportError("Mocked ImportError"))
        
        # Reload the module to simulate the import error
        with self.assertRaises(ImportError):
            # This should raise an ImportError because models.py can't be imported
            import importlib
            importlib.reload(sys.modules.get('LightRAG.reasoning', None))
            import LightRAG.reasoning


if __name__ == '__main__':
    unittest.main() 