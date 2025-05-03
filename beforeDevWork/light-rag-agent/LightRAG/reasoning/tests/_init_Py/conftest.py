"""
Pytest configuration for the reasoning module tests.

This module contains fixtures and configuration for pytest.
"""

import os
import sys
import pytest

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


@pytest.fixture
def mock_deepseek_model():
    """
    Fixture that provides a mock DeepSeekModel class.
    
    This can be used in tests to avoid actual API calls to DeepSeek R1.
    """
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