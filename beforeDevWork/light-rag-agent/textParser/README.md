# Text Converter Application

A Streamlit application for converting text files to well-structured paragraph format using DeepSeek AI.

## Features

- Convert traditional Chinese text files to well-structured paragraphs
- Real-time display of conversion results
- Progress tracking (current file, remaining files, execution time)
- API connection testing
- Debug mode for troubleshooting

## Setup

### Automatic Setup

Run the setup script to automatically create a virtual environment and install dependencies:

```bash
python setup.py
```

### Manual Setup

1. Create a virtual environment:

```bash
python -m venv venv
```

2. Activate the virtual environment:

- Windows:
```bash
venv\Scripts\activate
```

- macOS/Linux:
```bash
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project directory with your DeepSeek API key:

```
DEEPSEEK_API_KEY=your_api_key_here
```

## Usage

1. Run the Streamlit application:

```bash
streamlit run app.py
```

2. The application will open in your default web browser.

3. Click "Test API Connection" to verify your DeepSeek API key is working.

4. Click "Start Processing Files" to begin converting text files.

5. For each file, you can:
   - Review the converted text
   - Confirm and save the conversion
   - Cancel and skip to the next file
   - Regenerate the converted text

## Troubleshooting

- If you encounter issues, enable Debug Mode to view detailed logs
- Check the `debug.log` file for error messages
- Ensure your DeepSeek API key is correctly set in the `.env` file

## Directory Structure

```
textParser/
├── app.py             # Main application file
├── setup.py           # Setup script for environment setup
├── requirements.txt   # Python dependencies
├── .env               # Environment variables (create this file)
├── debug.log          # Log file (created when app runs)
└── README.md          # This file
``` 