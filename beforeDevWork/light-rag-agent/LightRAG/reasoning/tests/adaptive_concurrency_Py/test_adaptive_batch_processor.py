"""
Unit tests for the AdaptiveBatchProcessor class in adaptive_concurrency.py.

This test module verifies the functionality of the AdaptiveBatchProcessor class,
focusing on its ability to adaptively batch API calls based on system state.
"""

import unittest
import pytest
import time
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import os
import sys
import importlib

# Add parent directory to path if running standalone
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
grandparent_dir = os.path.dirname(parent_dir)
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)

# Import the module to test
from LightRAG.reasoning.adaptive_concurrency import AdaptiveBatchProcessor, AdaptiveConcurrencyManager


class TestAdaptiveBatchProcessor(unittest.TestCase):
    """Tests for the AdaptiveBatchProcessor class."""

    def setUp(self):
        """Set up the test case."""
        self.test_results_dir = os.path.join(parent_dir, "test_results")
        os.makedirs(self.test_results_dir, exist_ok=True)
        
        # Create a specific directory for this test module
        timestamp = importlib.import_module("datetime").datetime.now().strftime('%Y%m%d_%H%M%S')
        self.test_unit_dir = os.path.join(self.test_results_dir, f"test_adaptive_batch_processor_{timestamp}")
        os.makedirs(self.test_unit_dir, exist_ok=True)

        # Create a custom error logger for unittest (similar to pytest fixture)
        self.error_logger = self._create_error_logger()
        
        # Create a mock concurrency manager for all tests
        self.mock_concurrency_manager = MagicMock(spec=AdaptiveConcurrencyManager)
        self.mock_concurrency_manager.recommend_batch_size.return_value = 3
        self.mock_concurrency_manager.monitoring_active = False

    def _create_error_logger(self):
        """Create a local error logger function when not running under pytest."""
        def log_error(msg, exc=None):
            # Create a file for this test method
            method_name = self._testMethodName if hasattr(self, '_testMethodName') else 'unknown'
            log_file = os.path.join(self.test_unit_dir, f"TestAdaptiveBatchProcessor_{method_name}_results.txt")
            
            import traceback
            with open(log_file, 'a') as f:
                f.write(f"ERROR: {msg}\n")
                if exc:
                    f.write(f"Exception: {str(exc)}\n")
                    f.write(f"Traceback: {traceback.format_exc()}\n")
                f.write("-"*80 + "\n\n")
        
        return log_error

    def test_init(self):
        """Test case for expected use - initialization."""
        try:
            # Create processor with default values
            processor = AdaptiveBatchProcessor(self.mock_concurrency_manager)
            
            # Verify initialization
            self.assertEqual(processor.concurrency_manager, self.mock_concurrency_manager)
            self.assertEqual(processor.max_batch_size, 10)
            self.assertIsNone(processor.token_counter)
            
            # Verify monitoring started
            self.mock_concurrency_manager.start_monitoring.assert_called_once()
            
        except Exception as e:
            self.error_logger("Error in test_init", e)
            raise

    def test_batch_items(self):
        """Test case for expected use - batching items."""
        try:
            # Create processor
            processor = AdaptiveBatchProcessor(self.mock_concurrency_manager)
            
            # Create test items
            items = [f"item_{i}" for i in range(10)]
            
            # Setup mock
            self.mock_concurrency_manager.recommend_batch_size.return_value = 3
            
            # Batch items
            batches = processor.batch_items(items)
            
            # Verify batching
            self.assertEqual(len(batches), 4)  # Should create 4 batches with size 3,3,3,1
            self.assertEqual(len(batches[0]), 3)
            self.assertEqual(len(batches[1]), 3)
            self.assertEqual(len(batches[2]), 3)
            self.assertEqual(len(batches[3]), 1)
            
            # Verify all items included
            all_items = [item for batch in batches for item in batch]
            self.assertEqual(all_items, items)
            
            # Verify recommend_batch_size was called
            self.mock_concurrency_manager.recommend_batch_size.assert_called_once()
            
        except Exception as e:
            self.error_logger("Error in test_batch_items", e)
            raise

    def test_batch_items_empty(self):
        """Test edge case - batching empty items list."""
        try:
            # Create processor
            processor = AdaptiveBatchProcessor(self.mock_concurrency_manager)
            
            # Batch empty items
            batches = processor.batch_items([])
            
            # Verify result
            self.assertEqual(batches, [])
            
            # Verify recommend_batch_size was not called
            self.mock_concurrency_manager.recommend_batch_size.assert_not_called()
            
        except Exception as e:
            self.error_logger("Error in test_batch_items_empty", e)
            raise

    def test_batch_by_tokens(self):
        """Test case for expected use - batching by tokens."""
        try:
            # Create mock token counter
            def mock_token_counter(text):
                # Simple mock: 1 token per character
                return len(text)
            
            # Create processor with token counter
            processor = AdaptiveBatchProcessor(
                self.mock_concurrency_manager,
                token_counter=mock_token_counter
            )
            
            # Create test items with varying token counts
            # Each tuple is (text, data)
            items = [
                ("short", "data1"),       # 5 tokens
                ("medium text", "data2"), # 11 tokens
                ("very long text here", "data3"), # 19 tokens
                ("tiny", "data4"),        # 4 tokens
                ("another medium one", "data5"), # 18 tokens
            ]
            
            # Batch items by tokens with max 20 tokens per batch
            batches = processor.batch_by_tokens(items, max_tokens_per_batch=20)
            
            # Verify batching
            self.assertEqual(len(batches), 3)  # Should create 3 batches
            
            # First batch should have "short" and "medium text" (16 tokens total)
            self.assertEqual(len(batches[0]), 2)
            self.assertEqual(batches[0][0][0], "short")
            self.assertEqual(batches[0][1][0], "medium text")
            
            # Second batch should have "very long text here" (19 tokens)
            self.assertEqual(len(batches[1]), 1)
            self.assertEqual(batches[1][0][0], "very long text here")
            
            # Third batch should have "tiny" and "another medium one" (22 tokens, but can't split further)
            self.assertEqual(len(batches[2]), 2)
            self.assertEqual(batches[2][0][0], "tiny")
            self.assertEqual(batches[2][1][0], "another medium one")
            
        except Exception as e:
            self.error_logger("Error in test_batch_by_tokens", e)
            raise

    def test_batch_by_tokens_without_counter(self):
        """Test edge case - batching by tokens without a token counter."""
        try:
            # Create processor without token counter
            processor = AdaptiveBatchProcessor(self.mock_concurrency_manager)
            
            # Create test items
            items = [("text1", "data1"), ("text2", "data2"), ("text3", "data3")]
            
            # Setup mock
            self.mock_concurrency_manager.recommend_batch_size.return_value = 2
            
            # Batch items by tokens (should fall back to batch_items)
            batches = processor.batch_by_tokens(items, max_tokens_per_batch=100)
            
            # Verify batching (should be the same as batch_items)
            self.assertEqual(len(batches), 2)  # Should create 2 batches with size 2,1
            self.assertEqual(len(batches[0]), 2)
            self.assertEqual(len(batches[1]), 1)
            
            # Verify all items included
            all_items = [item for batch in batches for item in batch]
            self.assertEqual(all_items, items)
            
        except Exception as e:
            self.error_logger("Error in test_batch_by_tokens_without_counter", e)
            raise

    def test_process_batches(self):
        """Test case for expected use - processing batches."""
        try:
            # Create processor
            processor = AdaptiveBatchProcessor(self.mock_concurrency_manager)
            
            # Create test batches
            batches = [["item1", "item2"], ["item3", "item4", "item5"]]
            
            # Create mock process function
            def process_func(batch):
                # Mock process: append "_processed" to each item
                return [f"{item}_processed" for item in batch]
            
            # Process batches
            results = processor.process_batches(batches, process_func)
            
            # Verify results
            expected_results = ["item1_processed", "item2_processed", 
                              "item3_processed", "item4_processed", "item5_processed"]
            self.assertEqual(results, expected_results)
            
            # Verify record_api_call called for each batch
            self.assertEqual(self.mock_concurrency_manager.record_api_call.call_count, 2)
            
        except Exception as e:
            self.error_logger("Error in test_process_batches", e)
            raise

    def test_process_batches_empty(self):
        """Test edge case - processing empty batches."""
        try:
            # Create processor
            processor = AdaptiveBatchProcessor(self.mock_concurrency_manager)
            
            # Create mock process function
            def process_func(batch):
                return [f"{item}_processed" for item in batch]
            
            # Process empty batches
            results = processor.process_batches([], process_func)
            
            # Verify results
            self.assertEqual(results, [])
            
            # Verify record_api_call not called
            self.mock_concurrency_manager.record_api_call.assert_not_called()
            
        except Exception as e:
            self.error_logger("Error in test_process_batches_empty", e)
            raise

    def test_process_batches_error(self):
        """Test failure case - error during batch processing."""
        try:
            # Create processor
            processor = AdaptiveBatchProcessor(self.mock_concurrency_manager)
            
            # Create test batches
            batches = [["item1", "item2"], ["item3", "item4"]]
            
            # Create mock process function that raises an error on the second batch
            def process_func(batch):
                if "item3" in batch:
                    raise Exception("Simulated error in batch processing")
                return [f"{item}_processed" for item in batch]
            
            # Process batches (should handle the error and continue)
            results = processor.process_batches(batches, process_func)
            
            # Verify results - should only have results from first batch
            expected_results = ["item1_processed", "item2_processed"]
            self.assertEqual(results, expected_results)
            
            # Verify record_api_call called for both batches (success and failure)
            self.assertEqual(self.mock_concurrency_manager.record_api_call.call_count, 2)
            
            # Verify the second call recorded a failure
            args, kwargs = self.mock_concurrency_manager.record_api_call.call_args_list[1]
            self.assertEqual(kwargs["success"], False)
            
        except Exception as e:
            self.error_logger("Error in test_process_batches_error", e)
            raise

    @pytest.mark.asyncio
    async def test_process_batches_async(self):
        """Test case for expected use - processing batches asynchronously."""
        try:
            # Create processor
            processor = AdaptiveBatchProcessor(self.mock_concurrency_manager)
            
            # Create test batches
            batches = [["item1", "item2"], ["item3", "item4", "item5"]]
            
            # Create mock async process function
            async def process_func_async(batch):
                # Mock process: append "_processed" to each item
                await asyncio.sleep(0.01)  # Simulate async work
                return [f"{item}_processed" for item in batch]
            
            # Process batches asynchronously
            results = await processor.process_batches_async(batches, process_func_async)
            
            # Verify results
            expected_results = ["item1_processed", "item2_processed", 
                              "item3_processed", "item4_processed", "item5_processed"]
            self.assertEqual(results, expected_results)
            
            # Verify record_api_call called for each batch
            self.assertEqual(self.mock_concurrency_manager.record_api_call.call_count, 2)
            
        except Exception as e:
            self.error_logger("Error in test_process_batches_async", e)
            raise

    def test_shutdown(self):
        """Test case for expected use - shutdown."""
        try:
            # Create processor
            processor = AdaptiveBatchProcessor(self.mock_concurrency_manager)
            
            # Call shutdown
            processor.shutdown()
            
            # Verify concurrency manager's stop_monitoring was called
            self.mock_concurrency_manager.stop_monitoring.assert_called_once()
            
        except Exception as e:
            self.error_logger("Error in test_shutdown", e)
            raise


# Run pytest compatible tests
@pytest.fixture
def mock_concurrency_manager():
    """Fixture that provides a mock AdaptiveConcurrencyManager."""
    mock_manager = MagicMock(spec=AdaptiveConcurrencyManager)
    mock_manager.recommend_batch_size.return_value = 3
    mock_manager.monitoring_active = False
    return mock_manager

def test_batch_processor_init(mock_concurrency_manager, error_logger):
    """Test initialization of AdaptiveBatchProcessor with pytest fixtures."""
    try:
        # Create processor with default values
        processor = AdaptiveBatchProcessor(mock_concurrency_manager)
        
        # Verify initialization
        assert processor.concurrency_manager == mock_concurrency_manager
        assert processor.max_batch_size == 10
        assert processor.token_counter is None
        
        # Verify monitoring started
        mock_concurrency_manager.start_monitoring.assert_called_once()
        
    except Exception as e:
        error_logger("Error in test_batch_processor_init", e)
        raise

def test_batch_processor_batch_items(mock_concurrency_manager, error_logger):
    """Test batching items with pytest fixtures."""
    try:
        # Create processor
        processor = AdaptiveBatchProcessor(mock_concurrency_manager)
        
        # Create test items
        items = [f"item_{i}" for i in range(8)]
        
        # Setup mock
        mock_concurrency_manager.recommend_batch_size.return_value = 2
        
        # Batch items
        batches = processor.batch_items(items)
        
        # Verify batching
        assert len(batches) == 4  # Should create 4 batches with size 2 each
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 2
        assert len(batches[3]) == 2
        
        # Verify all items included
        all_items = [item for batch in batches for item in batch]
        assert all_items == items
        
    except Exception as e:
        error_logger("Error in test_batch_processor_batch_items", e)
        raise

@pytest.mark.asyncio
async def test_process_batches_async_with_error(mock_concurrency_manager, error_logger):
    """Test failure case - error during async batch processing."""
    try:
        # Create processor
        processor = AdaptiveBatchProcessor(mock_concurrency_manager)
        
        # Create test batches
        batches = [["item1", "item2"], ["item3", "item4"]]
        
        # Create mock async process function with error
        async def process_func_async(batch):
            await asyncio.sleep(0.01)  # Simulate async work
            if "item3" in batch:
                raise Exception("Simulated error in async batch processing")
            return [f"{item}_processed" for item in batch]
        
        # Process batches asynchronously
        results = await processor.process_batches_async(batches, process_func_async)
        
        # Verify results - should only have results from first batch
        expected_results = ["item1_processed", "item2_processed"]
        assert results == expected_results
        
        # Verify the second call recorded a failure
        args, kwargs = mock_concurrency_manager.record_api_call.call_args_list[1]
        assert kwargs["success"] is False
        assert kwargs["rate_limited"] is False
        
    except Exception as e:
        error_logger("Error in test_process_batches_async_with_error", e)
        raise


if __name__ == '__main__':
    unittest.main() 