# Advanced RAG System with Interactive UI

This repository contains an advanced Retrieval-Augmented Generation (RAG) system with an interactive UI built using Streamlit.

## Features

- Multi-format document processing (PDF, DOCX, TXT, HTML)
- Multi-level text chunking with configurable parameters
- Query expansion using OpenAI models
- Vector search with result comparison
- Interactive embedding space visualization

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Running the Application

Run the Streamlit app with:

```
streamlit run rag_ui.py
```

The application will open in your default web browser.

## Usage

1. **Upload a document** (PDF, Word, Text, or HTML) using the file uploader
2. **Process the document** by clicking the "Process Document" button
3. **Enter your question** in the query text box
4. Choose whether to use **Query Expansion** and **Visualization**
5. Click **Search** to see the results
6. Explore the comparison between original and expanded query results
7. View the embedding space visualization to understand document relationships

## Supported Document Types

- PDF documents (.pdf)
- Word documents (.docx)
- Plain text files (.txt)
- HTML files (.html, .htm)

## Components

- `expansion_answer.py`: Core RAG system with query expansion functionality
- `rag_ui.py`: Streamlit UI for the RAG system
- `helper_utils.py`: Utility functions for text processing and visualization
- `requirements.txt`: Dependencies for the project

## Advanced Configuration

You can adjust various parameters in the sidebar:
- OpenAI model selection for query expansion
- Character and token chunk sizes for text splitting
- Number of results to retrieve
- Debug mode for detailed logging 