"""
Unit tests for the reasoning module's agent.py.

This module tests the ReasoningStepLogger and ReasoningAgent classes,
which are responsible for tracking reasoning steps and implementing
core reasoning functionality for complex query analysis.
"""

import unittest
import sys
import os
import json
import time
import pytest
from unittest.mock import patch, MagicMock, mock_open

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)

# Import the module under test
from LightRAG.reasoning.agent import ReasoningStepLogger, ReasoningAgent
from LightRAG.reasoning.config import ReasoningError


class TestReasoningStepLogger(unittest.TestCase):
    """
    Tests for the ReasoningStepLogger class in agent.py
    """
    
    def setUp(self):
        """Set up the test case with a logger instance."""
        # Setup the test results directory
        self.test_results_dir = os.path.join(parent_dir, "test_results")
        os.makedirs(self.test_results_dir, exist_ok=True)
        
        # Create a directory for this test module
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.test_unit_dir = os.path.join(self.test_results_dir, f"test_agent_{timestamp}")
        os.makedirs(self.test_unit_dir, exist_ok=True)
        
        # Create a new logger for each test
        self.logger = ReasoningStepLogger()
        
        # Create a method to log errors
        self.log_error = self._create_error_logger()
    
    def _create_error_logger(self):
        """Create a local error logger function."""
        def log_error(msg, exc=None):
            # Create a file for this test method
            method_name = self._testMethodName if hasattr(self, '_testMethodName') else 'unknown'
            log_file = os.path.join(self.test_unit_dir, f"TestReasoningStepLogger_{method_name}_results.txt")
            
            with open(log_file, 'a') as f:
                f.write(f"ERROR: {msg}\n")
                if exc:
                    f.write(f"Exception: {str(exc)}\n")
                    import traceback
                    f.write(f"Traceback: {traceback.format_exc()}\n")
                f.write("-"*80 + "\n\n")
        
        return log_error
    
    def test_start_session_normal(self):
        """Test the normal case for starting a session."""
        try:
            # Test starting a session
            query = "What is the meaning of life?"
            session_id = self.logger.start_session(query)
            
            # Check that a session was started
            self.assertIsNotNone(session_id)
            self.assertEqual(self.logger.current_session_id, session_id)
            self.assertIsNotNone(self.logger.start_time)
            self.assertEqual(len(self.logger.steps), 1)
            
            # Check the initial step
            self.assertEqual(self.logger.steps[0]["step_name"], "session_start")
            self.assertEqual(self.logger.steps[0]["data"]["query"], query)
        except Exception as e:
            self.log_error("Error in test_start_session_normal", e)
            raise
    
    def test_log_step_with_large_data(self):
        """Test an edge case of logging a step with very large data."""
        try:
            # Start a session
            self.logger.start_session("Test query")
            
            # Create large data
            large_text = "x" * 1000  # 1000 character string
            large_list = list(range(100))  # List with 100 items
            
            # Log a step with large data
            self.logger.log_step("large_data_step", {
                "large_text": large_text,
                "large_list": large_list
            })
            
            # Verify the step was logged
            self.assertEqual(len(self.logger.steps), 2)  # Session start + our step
            step = self.logger.steps[1]
            
            # Check that large data was truncated
            self.assertEqual(len(step["data"]["large_text"]), 203)  # 200 chars + "..."
            self.assertEqual(len(step["data"]["large_list"]), 6)   # 5 items + ["..."]
            self.assertTrue(step["data"]["large_text"].endswith("..."))
        except Exception as e:
            self.log_error("Error in test_log_step_with_large_data", e)
            raise
    
    def test_log_step_without_session(self):
        """Test the edge case of logging a step without starting a session."""
        try:
            # Reset the logger
            self.logger = ReasoningStepLogger()
            
            # Log a step without starting a session
            self.logger.log_step("test_step", {"key": "value"})
            
            # Verify a session was automatically created
            self.assertIsNotNone(self.logger.current_session_id)
            self.assertEqual(len(self.logger.steps), 2)  # Auto-created session start + our step
        except Exception as e:
            self.log_error("Error in test_log_step_without_session", e)
            raise
    
    def test_log_step_failure(self):
        """Test the failure case when logging a step with problematic data."""
        try:
            # Start a session
            self.logger.start_session("Test query")
            
            # Create a circular reference that can't be JSON serialized
            problematic_data = {}
            problematic_data["self_reference"] = problematic_data  # Circular reference
            
            # Log a step with the problematic data - this should not crash
            self.logger.log_step("problematic_step", problematic_data)
            
            # Verify the step was logged despite the error
            self.assertEqual(len(self.logger.steps), 2)  # Session start + our step
        except Exception as e:
            self.log_error("Error in test_log_step_failure", e)
            raise
    
    def test_get_session_summary(self):
        """Test getting a summary of the session."""
        try:
            # Start a session
            self.logger.start_session("Test query")
            
            # Log a few steps
            self.logger.log_step("step1", {"data1": "value1"})
            self.logger.log_step("step2", {"data2": "value2"})
            
            # Get the summary
            summary = self.logger.get_session_summary()
            
            # Verify the summary
            self.assertEqual(summary["total_steps"], 3)  # Session start + 2 steps
            self.assertEqual(summary["steps"], ["session_start", "step1", "step2"])
        except Exception as e:
            self.log_error("Error in test_get_session_summary", e)
            raise
    
    def test_end_session(self):
        """Test ending a session."""
        try:
            # Start a session
            self.logger.start_session("Test query")
            
            # Log a step
            self.logger.log_step("test_step", {"key": "value"})
            
            # End the session
            steps = self.logger.end_session()
            
            # Verify the end session step was added
            self.assertEqual(len(steps), 3)  # Start + test_step + end
            self.assertEqual(steps[2]["step_name"], "session_end")
        except Exception as e:
            self.log_error("Error in test_end_session", e)
            raise
    
    def test_empty_session(self):
        """Test the edge case of ending an empty session."""
        try:
            # Reset the logger
            self.logger = ReasoningStepLogger()
            
            # End the session without starting it
            steps = self.logger.end_session()
            
            # Verify an empty list is returned
            self.assertEqual(len(steps), 0)
        except Exception as e:
            self.log_error("Error in test_empty_session", e)
            raise


class TestReasoningAgent(unittest.TestCase):
    """
    Tests for the ReasoningAgent class in agent.py
    """
    
    def setUp(self):
        """Set up the test case with a mocked ReasoningAgent."""
        # Setup the test results directory
        self.test_results_dir = os.path.join(parent_dir, "test_results")
        os.makedirs(self.test_results_dir, exist_ok=True)
        
        # Create a directory for this test module
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.test_unit_dir = os.path.join(self.test_results_dir, f"test_agent_{timestamp}")
        os.makedirs(self.test_unit_dir, exist_ok=True)
        
        # Create mocks
        self.mock_model = MagicMock()
        
        # Mock the model.call method
        def mock_call(prompt, options=None):
            if "analyze this question" in prompt.lower():
                return """
                Complexity level: Moderate
                Question type: Factoid
                Key concepts: Python, testing, unittest
                Whether the question requires breaking down into sub-questions: No
                Domain or knowledge areas: Programming, Software Testing
                """
            elif "integrate this information" in prompt.lower():
                return "This is an integrated answer based on the retrieved information."
            else:
                return "Mock response"
        
        self.mock_model.call.side_effect = mock_call
        
        # Create the agent with mocked dependencies
        self.agent = ReasoningAgent(model=self.mock_model)
        
        # Replace the ChainOfThought instance with a mock
        self.mock_cot = MagicMock()
        self.mock_cot.decompose_question.return_value = [
            {
                "id": 1,
                "question": "What is X?",
                "relevance": "Understanding X is important",
                "dependencies": []
            },
            {
                "id": 2,
                "question": "How does Y relate to X?",
                "relevance": "The relationship between X and Y is key",
                "dependencies": [1]
            }
        ]
        self.agent.cot = self.mock_cot
        
        # Create a method to log errors
        self.log_error = self._create_error_logger()
    
    def _create_error_logger(self):
        """Create a local error logger function."""
        def log_error(msg, exc=None):
            # Create a file for this test method
            method_name = self._testMethodName if hasattr(self, '_testMethodName') else 'unknown'
            log_file = os.path.join(self.test_unit_dir, f"TestReasoningAgent_{method_name}_results.txt")
            
            with open(log_file, 'a') as f:
                f.write(f"ERROR: {msg}\n")
                if exc:
                    f.write(f"Exception: {str(exc)}\n")
                    import traceback
                    f.write(f"Traceback: {traceback.format_exc()}\n")
                f.write("-"*80 + "\n\n")
        
        return log_error
    
    def test_analyze_query_normal(self):
        """Test the normal case for analyzing a query."""
        try:
            # Analyze a query
            query = "What is the best way to test Python code?"
            analysis = self.agent.analyze_query(query)
            
            # Verify the analysis
            self.assertIsNotNone(analysis)
            self.assertIn("complexity", analysis)
            self.assertIn("question_type", analysis)
            self.assertIn("key_concepts", analysis)
            
            # Verify model was called
            self.mock_model.call.assert_called_once()
            
            # Save the analysis for inspection
            result_file = os.path.join(self.test_unit_dir, "TestReasoningAgent_test_analyze_query_normal_results.txt")
            with open(result_file, 'w') as f:
                f.write(f"Query: {query}\n")
                f.write(f"Analysis: {json.dumps(analysis, indent=2)}\n")
                f.write("-"*80 + "\n\n")
        except Exception as e:
            self.log_error("Error in test_analyze_query_normal", e)
            raise
    
    def test_decompose_problem_simple(self):
        """Test problem decomposition for a simple query."""
        try:
            # Create a simple analysis
            analysis = {
                "complexity": "simple",
                "requires_decomposition": False
            }
            
            # Decompose a simple problem
            query = "What is Python?"
            sub_questions = self.agent.decompose_problem(query, analysis)
            
            # Verify the sub-questions - there should be only one (the original)
            self.assertEqual(len(sub_questions), 1)
            self.assertEqual(sub_questions[0]["question"], query)
            
            # Verify the CoT was not called
            self.mock_cot.decompose_question.assert_not_called()
            
            # Save the result for inspection
            result_file = os.path.join(self.test_unit_dir, "TestReasoningAgent_test_decompose_problem_simple_results.txt")
            with open(result_file, 'w') as f:
                f.write(f"Query: {query}\n")
                f.write(f"Analysis: {json.dumps(analysis, indent=2)}\n")
                f.write(f"Sub-questions: {json.dumps(sub_questions, indent=2)}\n")
                f.write("-"*80 + "\n\n")
        except Exception as e:
            self.log_error("Error in test_decompose_problem_simple", e)
            raise
    
    def test_decompose_problem_complex(self):
        """Test problem decomposition for a complex query."""
        try:
            # Create a complex analysis
            analysis = {
                "complexity": "complex",
                "requires_decomposition": True
            }
            
            # Decompose a complex problem
            query = "Compare and contrast different machine learning models for text classification"
            sub_questions = self.agent.decompose_problem(query, analysis)
            
            # Verify the sub-questions - should use the mocked CoT response
            self.assertEqual(len(sub_questions), 2)  # From our mocked CoT
            
            # Verify the CoT was called
            self.mock_cot.decompose_question.assert_called_once_with(query)
            
            # Save the result for inspection
            result_file = os.path.join(self.test_unit_dir, "TestReasoningAgent_test_decompose_problem_complex_results.txt")
            with open(result_file, 'w') as f:
                f.write(f"Query: {query}\n")
                f.write(f"Analysis: {json.dumps(analysis, indent=2)}\n")
                f.write(f"Sub-questions: {json.dumps(sub_questions, indent=2)}\n")
                f.write("-"*80 + "\n\n")
        except Exception as e:
            self.log_error("Error in test_decompose_problem_complex", e)
            raise
    
    def test_determine_strategy(self):
        """Test determining the strategy for a sub-question."""
        try:
            # Create a test subproblem
            subproblem = {
                "id": 1,
                "question": "Who invented Python?",
                "relevance": "Understanding the creator is important"
            }
            
            # Determine the strategy
            strategy = self.agent.determine_strategy(subproblem)
            
            # Verify the strategy
            self.assertIsNotNone(strategy)
            self.assertIn("top_k", strategy)
            self.assertIn("method", strategy)
            
            # Save the result for inspection
            result_file = os.path.join(self.test_unit_dir, "TestReasoningAgent_test_determine_strategy_results.txt")
            with open(result_file, 'w') as f:
                f.write(f"Subproblem: {json.dumps(subproblem, indent=2)}\n")
                f.write(f"Strategy: {json.dumps(strategy, indent=2)}\n")
                f.write("-"*80 + "\n\n")
        except Exception as e:
            self.log_error("Error in test_determine_strategy", e)
            raise
    
    def test_integrate_results(self):
        """Test integrating results from multiple sub-questions."""
        try:
            # Create test sub-questions and results
            sub_questions = [
                {
                    "id": "1",
                    "question": "What is X?",
                    "relevance": "Understanding X is important"
                },
                {
                    "id": "2",
                    "question": "How does Y relate to X?",
                    "relevance": "The relationship is key"
                }
            ]
            
            subproblem_results = {
                "1": {
                    "results": [
                        {"content": "X is a programming language", "source": "Wikipedia"}
                    ]
                },
                "2": {
                    "results": [
                        {"content": "Y is an extension of X", "source": "Documentation"}
                    ]
                }
            }
            
            # Integrate the results
            original_query = "Explain X and Y"
            integrated = self.agent.integrate_results(subproblem_results, original_query, sub_questions)
            
            # Verify the integration
            self.assertIsNotNone(integrated)
            self.assertIn("answer", integrated)
            
            # Verify model call
            self.mock_model.call.assert_called()
            
            # Save the result for inspection
            result_file = os.path.join(self.test_unit_dir, "TestReasoningAgent_test_integrate_results_results.txt")
            with open(result_file, 'w') as f:
                f.write(f"Original query: {original_query}\n")
                f.write(f"Sub-questions: {json.dumps(sub_questions, indent=2)}\n")
                f.write(f"Subproblem results: {json.dumps(subproblem_results, indent=2)}\n")
                f.write(f"Integrated result: {json.dumps(integrated, indent=2)}\n")
                f.write("-"*80 + "\n\n")
        except Exception as e:
            self.log_error("Error in test_integrate_results", e)
            raise
    
    def test_execute_reasoning(self):
        """Test the complete reasoning process."""
        try:
            # Execute reasoning
            query = "What are the best practices for writing unit tests in Python?"
            result = self.agent.execute_reasoning(query)
            
            # Verify the result
            self.assertIsNotNone(result)
            self.assertIn("query", result)
            self.assertIn("analysis", result)
            self.assertIn("sub_questions", result)
            self.assertIn("strategies", result)
            
            # Save the result for inspection
            result_file = os.path.join(self.test_unit_dir, "TestReasoningAgent_test_execute_reasoning_results.txt")
            with open(result_file, 'w') as f:
                f.write(f"Query: {query}\n")
                f.write(f"Result: {json.dumps(result, indent=2)}\n")
                f.write("-"*80 + "\n\n")
        except Exception as e:
            self.log_error("Error in test_execute_reasoning", e)
            raise
    
    def test_analyze_query_error(self):
        """Test the failure case when the model raises an error during analysis."""
        try:
            # Make the model call raise an exception
            self.mock_model.call.side_effect = Exception("Model API error")
            
            # Try to analyze a query, should raise a ReasoningError
            query = "What is the meaning of life?"
            with self.assertRaises(ReasoningError):
                self.agent.analyze_query(query)
            
            # Save the error for inspection
            result_file = os.path.join(self.test_unit_dir, "TestReasoningAgent_test_analyze_query_error_results.txt")
            with open(result_file, 'w') as f:
                f.write(f"Query: {query}\n")
                f.write(f"Expected error: ReasoningError due to model API error\n")
                f.write("-"*80 + "\n\n")
        except Exception as e:
            self.log_error("Error in test_analyze_query_error", e)
            raise


# Add pytest-compatible test functions to use fixtures directly
@pytest.mark.usefixtures("mock_model", "mock_cot", "error_logger")
def test_agent_with_fixtures(mock_model, mock_cot, error_logger):
    """Test using pytest fixtures to demonstrate error logging."""
    try:
        # Create an agent with our fixtures
        agent = ReasoningAgent(model=mock_model)
        agent.cot = mock_cot
        
        # Test analysis
        query = "What is the capital of France?"
        analysis = agent.analyze_query(query)
        
        # Verify the analysis
        assert "complexity" in analysis, "Analysis should include complexity"
        assert "question_type" in analysis, "Analysis should include question type"
        
        # Test problem decomposition
        sub_questions = agent.decompose_problem(query, analysis)
        assert len(sub_questions) > 0, "Should return at least one sub-question"
        
        # Test strategy determination
        strategy = agent.determine_strategy(sub_questions[0])
        assert "top_k" in strategy, "Strategy should include top_k parameter"
        
    except Exception as e:
        error_logger("Error in test_agent_with_fixtures", e)
        raise


@pytest.mark.usefixtures("error_logger")
def test_parse_analysis_edge_case(error_logger):
    """Test the edge case of parsing incomplete analysis responses."""
    try:
        # Create an agent with a real model
        agent = ReasoningAgent()
        
        # Test parsing an incomplete response
        incomplete_response = "Complexity level: Complex\nQuestion type:"
        analysis = agent._parse_analysis(incomplete_response)
        
        # Check that we get default values where missing
        assert analysis["complexity"] == "complex", "Should extract the complexity"
        assert analysis["question_type"] == "unknown", "Should use default for missing question type"
        assert len(analysis["key_concepts"]) == 0, "Should have empty key_concepts"
        
    except Exception as e:
        error_logger("Error in test_parse_analysis_edge_case", e)
        raise


@pytest.mark.usefixtures("error_logger")
def test_integrate_results_empty_case(error_logger):
    """Test the edge case of integrating empty results."""
    try:
        # Create an agent with a mock model that returns empty
        model = MagicMock()
        model.call.return_value = "No information found."
        agent = ReasoningAgent(model=model)
        
        # Test integrating empty results
        empty_results = {}
        original_query = "What is X?"
        
        integrated = agent.integrate_results(empty_results, original_query)
        
        # Check that integration still happens
        assert "answer" in integrated, "Should include an answer field"
        assert model.call.called, "Model should be called even with empty results"
        
    except Exception as e:
        error_logger("Error in test_integrate_results_empty_case", e)
        raise


if __name__ == '__main__':
    unittest.main() 