# LightRAG vs BasicRAG: Comparing RAG Implementations

This project demonstrates two different implementations of Retrieval-Augmented Generation (RAG) for answering questions about Pydantic AI using its documentation:

1. **BasicRAG**: A traditional RAG implementation using ChromaDB for vector storage and similarity search
2. **LightRAG**: An advanced, lightweight RAG implementation with enhanced knowledge graph capabilities

## Project Goal

The primary goal of this project is to showcase the power and efficiency of LightRAG compared to traditional RAG implementations. LightRAG offers several advantages:

- **Simplified API**: LightRAG provides a more streamlined API with fewer configuration parameters
- **Automatic Document Processing**: LightRAG handles document chunking and embedding automatically
- **Knowledge Graph Integration**: LightRAG leverages knowledge graph capabilities for improved context understanding
- **More Efficient Retrieval**: LightRAG's query mechanism provides more relevant results with less configuration

## Installation

### Prerequisites
- Python 3.11+
- OpenAI API key

### Setup

1. Clone this repository

2. Create a `.env` file in both the `BasicRAG` and `LightRAG` directories (or whichever you want to use) with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

3. Set up a virtual environment and install dependencies:

   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # Windows
   .\venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   
   # Install dependencies for LightRAG
   cd LightRAG
   pip install -r requirements.txt
   
   # In a new terminal with activated venv, install BasicRAG dependencies
   cd BasicRAG
   pip install -r requirements.txt
   ```

## Running the Implementations

### LightRAG (Most Powerful)

1. **Insert Documentation** (this will take a while - using full Pydantic AI docs as an example!):
   ```bash
   cd LightRAG
   python insert_pydantic_docs.py
   ```
   This will fetch the Pydantic AI documentation and process it using LightRAG's advanced document processing.

2. **Run the Agent**:
   ```bash
   python rag_agent.py --question "How do I create a Pydantic AI agent?"
   ```

3. **Run the Interactive Streamlit App**:
   ```bash
   streamlit run streamlit_app.py
   ```
   This provides a chat interface where you can ask questions about Pydantic AI.

### BasicRAG

1. **Insert Documentation** (this will take a while - using full Pydantic AI docs as an example!):
   ```bash
   cd BasicRAG
   python insert_pydantic_docs.py
   ```
   This will fetch and process the Pydantic AI documentation into ChromaDB with manual chunking.

2. **Run the Agent**:
   ```bash
   python rag_agent.py --question "How do I create a Pydantic AI agent?"
   ```
   You can customize the number of results from the vector DB with `--n-results 10`.

3. **Run the Interactive Streamlit App**:
   ```bash
   streamlit run streamlit_app.py
   ```

## Key Differences Between Implementations

### Document Processing
- **BasicRAG**: Manually splits documents into chunks with specified size and overlap, requiring careful tuning
- **LightRAG**: Automatically handles document processing with intelligent chunking

### Vector Storage
- **BasicRAG**: Uses ChromaDB directly with manual collection management
- **LightRAG**: Abstracts storage details behind a clean API with optimized defaults

### Query Mechanism
- **BasicRAG**: Requires specifying the number of results to return
- **LightRAG**: Uses a more sophisticated query mechanism with different modes (e.g., "naive" or "hybrid")

### Code Complexity
- **BasicRAG**: Requires more boilerplate code for setting up collections and processing documents
- **LightRAG**: Offers a more concise API with fewer lines of code needed

## Project Structure

### LightRAG
- `LightRAG/rag_agent.py`: Pydantic AI agent using LightRAG
- `LightRAG/insert_pydantic_docs.py`: Script to fetch and process documentation
- `LightRAG/streamlit_app.py`: Interactive web interface
- `LightRAG/reasoning/`: Reasoning agent module with Chain of Thought capabilities
  - `agent.py`: Reasoning agent implementation
  - `cot.py`: Chain of thought implementation
  - `models.py`: DeepSeek R1 model interface
  - `pipeline.py`: Reasoning pipeline coordination
  - `parallel.py`: Parallel processing for sub-questions
  - `visualization.py`: Reasoning process visualization
  - `settings.py`: Configuration management
  - `adaptive_concurrency.py`: Dynamic concurrency control
  - `priority_scheduler.py`: Prioritized sub-question processing

### BasicRAG
- `BasicRAG/rag_agent.py`: Pydantic AI agent using traditional RAG with ChromaDB
- `BasicRAG/insert_pydantic_docs.py`: Script for document processing with manual chunking
- `BasicRAG/utils.py`: Utility functions for ChromaDB operations
- `BasicRAG/streamlit_app.py`: Interactive web interface
- `BasicRAG/reasoning/`: Reasoning module integration for BasicRAG implementation

### Tests
- `tests/LightRAG/reasoning/`: Unit tests for reasoning components
  - Tests for expected use, edge cases, and failure scenarios

## Comparing Performance

To compare the performance of both implementations:

1. Run both Streamlit apps (in separate terminals)
2. Ask the same questions to both agents
3. Compare the quality and relevance of responses
4. Note the differences in response time and accuracy

LightRAG typically provides more contextually relevant answers with less configuration, demonstrating the advantages of its enhanced knowledge graph capabilities and optimized retrieval mechanisms.

## Documentation Standards

### Code Documentation

This project follows comprehensive documentation standards:

- **API Documentation**
  - Generated using Sphinx
  - Detailed explanations for all modules, classes, and methods
  - Usage examples and implementation notes
  - Type hints and parameter descriptions

- **Docstring Format**
  - All code uses Google-style docstring format
  - Classes include purpose, initialization parameters, and usage notes
  - Methods specify parameters, return values, raises, and examples

Example docstring format:
```python
def process_query(query: str, options: Dict = None) -> Dict:
    """
    Process a user query through the reasoning pipeline.
    
    Args:
        query: The user's natural language question
        options: Optional configuration parameters
            
    Returns:
        Dict containing the processed results with reasoning steps
            
    Raises:
        ValueError: If query is empty or invalid
        APIError: If external API call fails
            
    Example:
        >>> result = process_query("How do I create a Pydantic model?")
        >>> print(result["answer"])
    """
```

### User Documentation

- **System Architecture**
  - Component diagrams and interaction flows
  - Subsystem descriptions and responsibilities
  - Data flow and processing pipeline details
  
  The system architecture documentation includes:
  
  1. **Core Components**
     - Detailed architecture diagrams of LightRAG and BasicRAG
     - Component interaction maps showing data flow between modules
     - Subsystem responsibility matrices with clear boundaries
     - Technology stack documentation for each component
  
  2. **Integration Points**
     - API documentation for connecting with external systems
     - Interface definitions for all public modules
     - Extension points for custom component development
     - Integration patterns and best practices
  
  3. **Data Flow**
     - End-to-end query processing pipeline visualization
     - Document processing and embedding workflows
     - Knowledge graph construction and maintenance processes
     - Retrieval and ranking mechanisms with detailed explanations

- **Installation and Configuration**
  - Detailed environment setup instructions
  - Configuration options and customization
  - Troubleshooting common installation issues
  
  Complete installation documentation provides:
  
  1. **Environment Setup Guides**
     - Detailed requirements for development, testing, and production
     - Step-by-step installation procedures for different operating systems
     - Docker containerization options with sample configurations
     - Virtual environment setup with dependency management
  
  2. **Configuration Reference**
     - Comprehensive settings documentation with examples
     - Environment variable reference with valid options
     - Configuration file templates with annotations
     - Optimization guides for different hardware configurations
  
  3. **Troubleshooting**
     - Common installation issues with resolutions
     - Dependency conflict resolution strategies
     - Environment validation tools and diagnostics
     - Installation verification procedures

- **Usage Tutorials**
  - Quick start guides for common use cases
  - Step-by-step tutorials for advanced features
  - Best practices for optimal results
  
  The project includes comprehensive tutorials:
  
  1. **Basic RAG System Tutorial**
     - Complete walkthrough for setting up BasicRAG
     - Guide for configuring ChromaDB collections
     - Examples of document chunking strategies
     - Demonstrations of query optimization techniques
  
  2. **LightRAG System Tutorial**
     - Step-by-step guide for implementing LightRAG
     - Instructions for knowledge graph integration
     - Examples of hybrid retrieval patterns
     - Guide for customizing retrieval parameters
  
  3. **Reasoning Agent Integration**
     - Tutorial for connecting reasoning modules
     - Guide for implementing Chain of Thought processes
     - Examples of sub-question generation and dependency tracking
     - Instructions for visualizing reasoning traces
  
  4. **Performance Optimization**
     - Guide for diagnosing retrieval bottlenecks
     - Step-by-step process for optimizing query latency
     - Instructions for implementing parallel processing
     - Examples of caching strategies for frequent queries
  
  5. **Interactive Web Interface**
     - Tutorial for customizing the Streamlit application
     - Guide for implementing user feedback collection
     - Instructions for adding visualization components
     - Examples of interface customization for specific use cases
     
  6. **Reasoning Module Visualization Interface**
     - **Running the Interface**: 
       ```bash
       cd LightRAG
       streamlit run reasoning/streamlit_app.py
       ```
     - **Interface Features**:
       - Complex question decomposition visualization
       - Chain of thought reasoning trace exploration
       - Dependency graph visualization of sub-questions
       - Step-by-step reasoning process exploration
       
     - **Using the Reasoning Interface**:
       1. Enter a complex question in the text area
       2. Click "開始推理分析" to start the reasoning process
       3. Explore the question analysis, showing complexity and type
       4. View the decomposed sub-questions and their dependencies
       5. Examine the retrieval results for each sub-question
       6. See the final integrated answer
       7. Download the complete reasoning trace as JSON for analysis
       
     - **Example Questions**:
       - "試舉例說明近百年間陸續出土的簡帛文獻，為群經與諸子揭露何許重要的學術訊息？"
       - "自元代將四書納入科舉以來，其地位幾乎與五經平起平坐。然而，其中《論語》和《孟子》二書歷代學者是否將其視為「經」，卻有不一樣的描述。請以漢、唐、宋等三個時期，舉出立證，比較二書入經過程之異同。 "
       - "試參考以下四則文字，據以評述各家史書(宜就己意加以闡說，不得流為語譯)。
(1) 苟悅省前漢之繁而為漢紀袁宏剪後漢之穢而為編年
(2) 晉史始有十八家之制作而成於唐臣之纂錄然好採詭異語多駢儷
(3) 李延壽之南北史司馬公喜其叙事簡勁賢於正史但恨其不作志書使制度不見耳
(4) 歐陽修作五代史立例精密取法春秋文簡而能暢事增而不費其為論必以鳴呼發之蓋以亂世之書故致其慨嘆之意也"

- **Frequently Asked Questions**
  - Common issues and solutions
  - Performance optimization tips
  - Integration guidance with other systems
  
  The FAQ section covers:
  
  1. **Implementation Questions**
     - When to choose LightRAG vs BasicRAG for specific use cases
     - How to integrate the system with existing applications
     - Methods for extending the system with custom components
     - Strategies for handling specialized document types
  
  2. **Performance Considerations**
     - Optimizing response time for large document collections
     - Balancing accuracy and speed in retrieval operations
     - Scaling strategies for high-volume query processing
     - Memory optimization techniques for resource-constrained environments
  
  3. **Troubleshooting Guide**
     - Resolving common retrieval quality issues
     - Debugging reasoning agent problems
     - Fixing integration errors with vector databases
     - Addressing API connection failures and rate limiting

### Documentation Files

The project's documentation is organized across several key files:

- **README.md**: Project overview, installation, usage, and feature documentation
- **PLANNING.md**: Architecture, design decisions, and project planning details
- **rag_reasoningAgent_devGuide.md**: Developer guide for understanding and extending the reasoning agent
- **Module-level documentation**: Each module has its own README explaining purpose and usage

### Documentation Generation

API documentation is automatically generated from codebase using Sphinx:

1. Documentation is built from docstrings and structured comments
2. Each module, class, and method includes detailed explanations
3. Usage examples are provided for key functionality
4. Notes on implementation decisions are included where relevant

### Testing Documentation

The project includes comprehensive testing documentation:

- **Test Structure**
  - Each component has corresponding test files in the tests/ directory
  - Test files follow naming convention `test_<module_name>.py`
  - Tests are organized to mirror the project structure

- **Test Coverage Requirements**
  - All tests ensure three key aspects:
    1. Expected use cases function correctly
    2. Edge cases are handled appropriately
    3. Failure scenarios are handled gracefully

- **Running Tests**
  - Detailed instructions for running specific tests
  - Support for test filtering and parameterization
  - Configuration for test reporting and coverage analysis

- **Test Development Guidelines**
  - Guidelines for mock usage to avoid external dependencies
  - Best practices for pytest fixture utilization
  - Standards for test naming and organization

## Contributing

Contributions to this project are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following the project's code style
4. Add tests for your changes
5. Update documentation to reflect your changes
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

Please ensure your code follows the project's documentation standards and includes appropriate tests.
