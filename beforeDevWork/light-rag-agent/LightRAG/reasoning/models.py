import os
import json
import time
import requests
from typing import Dict, List, Optional, Union, Any, Tuple
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from .config import (
    logger, 
    DeepSeekAPIError, 
    ReasoningError, 
    API_TIMEOUT, 
    MAX_RETRIES, 
    RETRY_BACKOFF,
    get_api_key,
    get_api_base,
    get_model_name
)

# Load environment variables
load_dotenv()

class DeepSeekModel:
    """
    DeepSeek R1 model interface encapsulation.
    
    This class provides methods to interact with the DeepSeek R1 API for 
    generating responses, chain of thought reasoning, and batch processing.
    
    Attributes:
        api_key (str): DeepSeek API key.
        api_base (str): Base URL for the DeepSeek API.
        model_name (str): Name of the DeepSeek model to use.
        default_options (Dict): Default parameters for API calls.
        session (requests.Session): Session for HTTP requests with retry mechanism.
        connection_status (Dict): Information about the API connection status.
    """
    
    def __init__(
        self, 
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        test_connection_on_init: bool = True
    ):
        """
        Initialize the DeepSeek R1 model interface.
        
        Args:
            model_name (Optional[str]): Name of the DeepSeek model to use.
            api_key (Optional[str]): DeepSeek API key. If None, reads from environment variable.
            api_base (Optional[str]): Base URL for the DeepSeek API. If None, uses default.
            test_connection_on_init (bool): Whether to test the connection on initialization.
        """
        self.api_key = api_key or get_api_key()
        self.api_base = api_base or get_api_base()
        self.model_name = model_name or get_model_name()
        self.default_options = {
            "temperature": 0.2,
            "max_tokens": 1024,
            "top_p": 0.95,
            "stream": False
        }
        
        # Connection status tracking
        self.connection_status = {
            "last_check_time": None,
            "is_connected": False,
            "last_error": None,
            "total_requests": 0,
            "failed_requests": 0
        }
        
        # Set up session with retry mechanism
        self.session = self._create_session()
        
        # Test connection on initialization if requested
        if test_connection_on_init:
            try:
                self.test_connection()
                logger.info(f"Successfully connected to DeepSeek API using model: {self.model_name}")
            except Exception as e:
                logger.warning(f"Connection test to DeepSeek API failed: {e}")
                # We don't raise here to allow for delayed initialization or offline usage
    
    def _create_session(self) -> requests.Session:
        """
        Create a requests session with retry mechanism.
        
        Returns:
            requests.Session: Configured session object.
        """
        session = requests.Session()
        retries = Retry(
            total=MAX_RETRIES,
            backoff_factor=RETRY_BACKOFF,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount('https://', adapter)
        session.mount('http://', adapter)
        return session
    
    def test_connection(self, detailed: bool = False) -> Union[bool, Dict[str, Any]]:
        """
        Test the connection to the DeepSeek API.
        
        Args:
            detailed (bool): If True, returns detailed diagnostics instead of just a boolean.
            
        Returns:
            Union[bool, Dict[str, Any]]: True if connection is successful or diagnostic information.
            
        Raises:
            DeepSeekAPIError: If connection test fails and detailed=False.
        """
        simple_prompt = "Hello, this is a connection test."
        start_time = time.time()
        
        try:
            # Try a simple API call with minimal tokens
            response = self.call(simple_prompt, {"max_tokens": 5})
            elapsed_time = time.time() - start_time
            
            # Update connection status
            self.connection_status = {
                "last_check_time": time.time(),
                "is_connected": True,
                "last_error": None,
                "response_time": elapsed_time,
                "total_requests": self.connection_status.get("total_requests", 0) + 1,
                "failed_requests": self.connection_status.get("failed_requests", 0)
            }
            
            # Return either boolean or detailed diagnostics
            if detailed:
                return {
                    "connected": True,
                    "response_time": elapsed_time,
                    "model": self.model_name,
                    "api_base": self.api_base,
                    "response_content": response[:50] + "..." if len(response) > 50 else response
                }
            return True
            
        except Exception as e:
            # Update connection status on failure
            self.connection_status = {
                "last_check_time": time.time(),
                "is_connected": False,
                "last_error": str(e),
                "total_requests": self.connection_status.get("total_requests", 0) + 1,
                "failed_requests": self.connection_status.get("failed_requests", 0) + 1
            }
            
            # Return either raise exception or return detailed diagnostics
            if detailed:
                return {
                    "connected": False,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "model": self.model_name,
                    "api_base": self.api_base,
                    "elapsed_time": time.time() - start_time
                }
            raise DeepSeekAPIError(f"Connection test failed: {str(e)}")
    
    def get_connection_diagnostics(self) -> Dict[str, Any]:
        """
        Get detailed diagnostics about the API connection.
        
        Returns:
            Dict[str, Any]: Diagnostic information.
        """
        # If we've never checked the connection, do it now
        if self.connection_status.get("last_check_time") is None:
            try:
                self.test_connection(detailed=True)
            except:
                pass
                
        # Calculate connection health metrics
        total_req = self.connection_status.get("total_requests", 0)
        failed_req = self.connection_status.get("failed_requests", 0)
        success_rate = 0 if total_req == 0 else (total_req - failed_req) / total_req * 100
            
        return {
            **self.connection_status,
            "success_rate": success_rate,
            "api_base": self.api_base,
            "model": self.model_name
        }
    
    def call(self, prompt: str, options: Optional[Dict] = None) -> str:
        """
        Call the DeepSeek R1 model with a single prompt.
        
        Args:
            prompt (str): The input prompt for the model.
            options (Optional[Dict]): Additional parameters to override defaults.
            
        Returns:
            str: The model's response text.
            
        Raises:
            DeepSeekAPIError: If the API call fails.
        """
        options = options or {}
        call_options = {**self.default_options, **options}
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            **call_options
        }
        
        logger.debug(f"Calling DeepSeek API with model {self.model_name}")
        
        try:
            start_time = time.time()
            response = self.session.post(
                f"{self.api_base}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=API_TIMEOUT
            )
            elapsed_time = time.time() - start_time
            
            # Log timing information
            logger.debug(f"DeepSeek API call completed in {elapsed_time:.2f} seconds")
            
            # Update connection metrics on success
            self.connection_status["total_requests"] = self.connection_status.get("total_requests", 0) + 1
            
            # Handle error responses
            if response.status_code != 200:
                error_data = {}
                try:
                    error_data = response.json()
                except:
                    pass
                
                # Update failed request count
                self.connection_status["failed_requests"] = self.connection_status.get("failed_requests", 0) + 1
                self.connection_status["last_error"] = f"API error {response.status_code}"
                
                logger.error(f"DeepSeek API error: Status {response.status_code}, Response: {error_data}")
                raise DeepSeekAPIError(
                    f"DeepSeek API returned error response",
                    status_code=response.status_code,
                    response=error_data
                )
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
        
        except requests.RequestException as e:
            # Update failed request metrics
            self.connection_status["failed_requests"] = self.connection_status.get("failed_requests", 0) + 1
            self.connection_status["last_error"] = str(e)
            
            logger.error(f"Request exception calling DeepSeek API: {e}")
            raise DeepSeekAPIError(f"Network error calling DeepSeek API: {str(e)}")
        
        except json.JSONDecodeError as e:
            # Update failed request metrics
            self.connection_status["failed_requests"] = self.connection_status.get("failed_requests", 0) + 1
            self.connection_status["last_error"] = str(e)
            
            logger.error(f"JSON decode error from DeepSeek API response: {e}")
            raise DeepSeekAPIError(f"Failed to parse DeepSeek API response: {str(e)}")
        
        except KeyError as e:
            # Update failed request metrics
            self.connection_status["failed_requests"] = self.connection_status.get("failed_requests", 0) + 1
            self.connection_status["last_error"] = str(e)
            
            logger.error(f"Unexpected response format from DeepSeek API: {e}")
            raise DeepSeekAPIError(f"Unexpected response format from DeepSeek API: {str(e)}")
        
        except Exception as e:
            # Update failed request metrics
            self.connection_status["failed_requests"] = self.connection_status.get("failed_requests", 0) + 1
            self.connection_status["last_error"] = str(e)
            
            logger.error(f"Unexpected error calling DeepSeek API: {e}")
            raise DeepSeekAPIError(f"Unexpected error calling DeepSeek API: {str(e)}")
    
    def generate_chain_of_thought(self, query: str) -> List[str]:
        """
        Generate chain of thought reasoning steps for a complex query.
        
        This method prompts the model to break down a complex problem into
        step-by-step reasoning steps, encouraging thorough analysis.
        
        Args:
            query (str): The complex query to analyze.
            
        Returns:
            List[str]: A list of reasoning steps as strings.
            
        Raises:
            ReasoningError: If the reasoning process fails.
        """
        prompt = f"""
        I need to solve this complex problem: "{query}"
        
        Please help me break this down into a chain of thought reasoning process.
        First, analyze what the question is asking.
        Then, identify the key components or sub-problems that need to be addressed.
        For each sub-problem, outline a reasoning approach.
        Finally, explain how these components should be combined to answer the original query.
        
        Format your response as a numbered list of distinct reasoning steps.
        """
        
        try:
            logger.debug(f"Generating chain of thought for query: {query[:50]}...")
            response = self.call(prompt, {"temperature": 0.3})
            
            # Process the response into a list of reasoning steps
            steps = []
            for line in response.split("\n"):
                line = line.strip()
                if line and (line[0].isdigit() or any(line.startswith(prefix) for prefix in ["Step ", "- "])):
                    # Remove numbering or bullet points
                    cleaned_line = line
                    for prefix in ["Step ", "- "]:
                        if cleaned_line.startswith(prefix):
                            cleaned_line = cleaned_line[len(prefix):]
                            break
                    if cleaned_line and cleaned_line[0].isdigit() and "." in cleaned_line:
                        # Remove "1." style numbering
                        parts = cleaned_line.split(".", 1)
                        if len(parts) > 1 and parts[0].isdigit():
                            cleaned_line = parts[1].strip()
                    
                    if cleaned_line:
                        steps.append(cleaned_line)
            
            if not steps:
                # If parsing failed, just split by newlines as fallback
                steps = [step.strip() for step in response.split("\n") if step.strip()]
            
            logger.debug(f"Generated {len(steps)} chain of thought steps")
            return steps
            
        except Exception as e:
            logger.error(f"Error generating chain of thought: {e}")
            raise ReasoningError(
                f"Failed to generate chain of thought reasoning: {str(e)}",
                step="generate_chain_of_thought",
                data={"query": query}
            )
    
    def batch_call(self, prompts: List[str], options: Optional[Dict] = None) -> List[str]:
        """
        Batch call the DeepSeek R1 model with multiple prompts.
        
        Args:
            prompts (List[str]): List of input prompts.
            options (Optional[Dict]): Additional parameters to override defaults.
            
        Returns:
            List[str]: List of model responses corresponding to each prompt.
            
        Raises:
            DeepSeekAPIError: If any of the API calls fail.
        """
        logger.debug(f"Batch calling DeepSeek API with {len(prompts)} prompts")
        results = []
        errors = []
        
        for i, prompt in enumerate(prompts):
            try:
                result = self.call(prompt, options)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in batch call for prompt {i}: {e}")
                errors.append((i, str(e)))
                results.append(None)  # Add None for failed calls
        
        if errors:
            error_msg = f"Batch call had {len(errors)} errors out of {len(prompts)} calls"
            logger.warning(error_msg)
            # Log details but don't raise if we have some successful results
            if len(errors) == len(prompts):
                raise DeepSeekAPIError(f"All batch calls failed: {error_msg}")
        
        return results  
