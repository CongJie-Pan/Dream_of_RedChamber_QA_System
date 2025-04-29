import os
import subprocess
import sys
import platform
from pathlib import Path

def print_step(message):
    """Print a step message with formatting."""
    print(f"\n{'='*80}\n{message}\n{'='*80}")

def create_virtual_environment():
    """Create a virtual environment."""
    print_step("Creating virtual environment...")
    
    if os.path.exists("venv"):
        print("Virtual environment already exists. Skipping creation.")
        return True
    
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("Virtual environment created successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating virtual environment: {e}")
        return False

def install_dependencies(venv_path):
    """Install dependencies from requirements.txt."""
    print_step("Installing dependencies...")
    
    # Determine the pip executable path based on the OS
    if platform.system() == "Windows":
        pip_exe = os.path.join(venv_path, "Scripts", "pip")
    else:
        pip_exe = os.path.join(venv_path, "bin", "pip")
    
    try:
        subprocess.run([pip_exe, "install", "--upgrade", "pip"], check=True)
        subprocess.run([pip_exe, "install", "-r", "requirements.txt"], check=True)
        print("Dependencies installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False

def create_env_file():
    """Create a .env file if it doesn't exist."""
    print_step("Creating .env file...")
    
    env_file = Path(".env")
    
    if env_file.exists():
        print(".env file already exists. Skipping creation.")
        return
    
    try:
        with open(env_file, "w") as f:
            f.write("DEEPSEEK_API_KEY=your_deepseek_api_key_here\n")
        
        print(".env file created. Please edit it to add your DeepSeek API key.")
    except Exception as e:
        print(f"Error creating .env file: {e}")

def main():
    """Main setup function."""
    print_step("Setting up Text Converter Application")
    
    # Create virtual environment
    venv_created = create_virtual_environment()
    if not venv_created:
        print("Failed to create virtual environment. Setup incomplete.")
        return
    
    # Install dependencies
    deps_installed = install_dependencies("venv")
    if not deps_installed:
        print("Failed to install dependencies. Setup incomplete.")
        return
    
    # Create .env file
    create_env_file()
    
    print_step("Setup complete!")
    
    # Print activation instructions
    if platform.system() == "Windows":
        activate_cmd = "venv\\Scripts\\activate"
    else:
        activate_cmd = "source venv/bin/activate"
    
    print(f"""
To activate the virtual environment:
    {activate_cmd}

To run the application:
    streamlit run app.py

Before running, make sure to edit the .env file with your DeepSeek API key.
""")

if __name__ == "__main__":
    main() 