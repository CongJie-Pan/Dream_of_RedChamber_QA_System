import fitz
import os
import re
from pathlib import Path

def extract_pdf_to_markdown(pdf_path, output_path=None):
    """
    Extract content from a PDF file and save it as markdown format.
    
    Args:
        pdf_path (str): Path to the PDF file
        output_path (str, optional): Path to save the output markdown file
    
    Returns:
        str: The extracted markdown content
    """
    print(f"Processing PDF: {pdf_path}")
    
    # Open the PDF
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return ""
    
    print(f"PDF has {len(doc)} pages")
    
    # Extract text from each page
    markdown_text = []
    
    # Get PDF filename without extension to use as title
    pdf_filename = os.path.basename(pdf_path)
    title = os.path.splitext(pdf_filename)[0]
    
    # Add title as heading
    markdown_text.append(f"# {title}\n\n")
    
    for page_num, page in enumerate(doc):
        print(f"Processing page {page_num + 1}/{len(doc)}")
        
        # Extract text
        text = page.get_text()
        
        # Clean up text (remove excessive newlines, etc.)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Potential section headings (lines that are shorter and may be titles)
        lines = text.split('\n')
        processed_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Check if this might be a heading (shorter line)
            if 3 <= len(line) <= 30 and (i == 0 or not lines[i-1].strip()):
                processed_lines.append(f"## {line}\n")
            else:
                processed_lines.append(line)
        
        # Join lines back with newlines, and add to markdown text
        page_text = '\n'.join(processed_lines)
        markdown_text.append(page_text)
        markdown_text.append("\n---\n")  # Page separator
    
    # Join all pages
    markdown_content = '\n'.join(markdown_text)
    
    # If output path is not specified, create one based on input filename
    if output_path is None:
        pdf_dir = os.path.dirname(pdf_path)
        output_path = os.path.join(pdf_dir, f"{title}.txt")
    
    # Save to file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        print(f"Markdown content saved to: {output_path}")
    except Exception as e:
        print(f"Error saving markdown content: {e}")
    
    return markdown_content

if __name__ == "__main__":
    # Input PDF path
    pdf_path = r"D:\AboutCoding\PKUDH_Project\DreamOf_RedMansions\QA_System\beforeDevWork\light-rag-agent\LightRAG\data\史學-政書體：三《通》詳述.pdf"
    
    # Extract and save as markdown
    extract_pdf_to_markdown(pdf_path) 