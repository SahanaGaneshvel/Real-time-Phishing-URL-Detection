#!/usr/bin/env python3
"""
Simplified startup script for Phishing URL Detection System
Starts both backend and frontend with a single command
"""

import os
import sys
import subprocess
import time
import threading
import webbrowser
from pathlib import Path

# Project / virtual environment paths
PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / "venv"
IS_WINDOWS = os.name == "nt"
VENV_BIN_DIR = VENV_DIR / ("Scripts" if IS_WINDOWS else "bin")
VENV_PYTHON = VENV_BIN_DIR / ("python.exe" if IS_WINDOWS else "python")

def _resolve_python_executable() -> Path:
    """Pick the Python executable to use for subprocesses."""
    if VENV_PYTHON.exists():
        print(f"✅ Using project virtual environment python: {VENV_PYTHON}")
        return VENV_PYTHON
    print("⚠️ Project virtual environment not found. Falling back to current interpreter.")
    return Path(sys.executable)

PYTHON_EXECUTABLE = _resolve_python_executable()
USING_PROJECT_VENV = PYTHON_EXECUTABLE == VENV_PYTHON

def _build_subprocess_env():
    """Ensure subprocesses inherit the virtual environment when available."""
    env = os.environ.copy()
    if USING_PROJECT_VENV:
        env["VIRTUAL_ENV"] = str(VENV_DIR)
        env["PATH"] = str(VENV_BIN_DIR) + os.pathsep + env.get("PATH", "")
    return env

SUBPROCESS_ENV = _build_subprocess_env()

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    from pathlib import Path
    env_path = Path('.env')
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print(f"✅ Loaded .env from: {env_path.absolute()}")
    else:
        # Try loading from current directory
        load_dotenv()
        print(f"⚠️ .env file not found at: {env_path.absolute()}")
        print(f"⚠️ Using default environment or system environment variables")
except ImportError:
    print("⚠️ python-dotenv not installed. Install with: pip install python-dotenv")

def check_environment_variables():
    """Check if required environment variables are set"""
    print("Checking environment variables...")
    
    # Check for .env file
    env_file = Path('.env')
    if env_file.exists():
        print("✓ .env file found")
    else:
        print("⚠ .env file not found")
        print("Please create a .env file based on env_template.txt")
    
    # Template/placeholder keys to reject (only obvious placeholders)
    template_safe_browsing_keys = [
        'your_safe_browsing_api_key_here',
        'YOUR_SAFE_BROWSING_API_KEY_HERE',
        ''
    ]
    template_gemini_keys = [
        'your_gemini_api_key_here',
        'YOUR_GEMINI_API_KEY_HERE',
        ''
    ]
    
    # Check Google Safe Browsing API key
    safe_browsing_key = os.getenv('GOOGLE_SAFE_BROWSING_API_KEY', '').strip()
    is_valid_safe_browsing = (
        safe_browsing_key and 
        safe_browsing_key not in template_safe_browsing_keys and
        len(safe_browsing_key) > 10
    )
    if is_valid_safe_browsing:
        print("✓ Google Safe Browsing API key: Configured")
    else:
        print("⚠ Google Safe Browsing API key: Not configured or using template key")
        print("   (Optional - system will work without it)")
        print("   Please set GOOGLE_SAFE_BROWSING_API_KEY in your .env file")
        print("   Get your API key from: https://console.developers.google.com/")
    
    # Check Gemini API key
    gemini_key = os.getenv('GEMINI_API_KEY', '').strip()
    is_valid_gemini = (
        gemini_key and 
        gemini_key not in template_gemini_keys and
        len(gemini_key) > 10
    )
    if is_valid_gemini:
        print("✓ Gemini API key: Configured")
    else:
        print("⚠ Gemini API key: Not configured or using template key")
        print("   (Chatbot will use fallback responses without it)")
        print("   Please set GEMINI_API_KEY in your .env file")
        print("   Get your API key from: https://makersuite.google.com/app/apikey")
    
    print()

def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")
    
    # Check core Python dependencies
    try:
        import flask, sklearn, pandas, numpy, joblib
        print("✓ Core Python dependencies: OK")
    except ImportError as e:
        print(f"✗ Missing core Python dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    # Check RAG dependencies (make it non-blocking)
    try:
        import faiss
        print("✓ FAISS: OK")
    except ImportError as e:
        print(f"⚠ FAISS missing: {e}")
    
    try:
        import langchain_community
        print("✓ LangChain: OK")
    except ImportError as e:
        print(f"⚠ LangChain missing: {e}")
    
    # Check sentence-transformers separately (it depends on torch which can have DLL issues)
    try:
        import sentence_transformers
        print("✓ Sentence Transformers: OK")
    except (ImportError, OSError) as e:
        print(f"⚠ Sentence Transformers: Not available ({type(e).__name__})")
        print("   Note: This is optional - RAG will use simple embeddings instead")
        if "DLL" in str(e) or "torch" in str(e).lower():
            print("   Tip: PyTorch DLL issue detected. RAG will still work with fallback embeddings.")
    
    # Check Gemini AI (required for chatbot)
    try:
        import google.generativeai
        print("✓ Gemini AI: Available")
    except ImportError:
        print("⚠ Gemini AI: Not installed (required for chatbot)")
        print("Please install: pip install google-generativeai")
    
    # Check Node.js
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Node.js: OK")
        else:
            print("✗ Node.js not found")
            return False
    except FileNotFoundError:
        print("✗ Node.js not found")
        return False
    
    # Check npm
    try:
        result = subprocess.run(['npm', '--version'], capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            print("✓ npm: OK")
        else:
            print("✗ npm not found")
            return False
    except FileNotFoundError:
        print("✗ npm not found")
        return False
    
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\nInstalling dependencies...")
    
    # Check if requirements.txt exists
    requirements_file = Path('requirements.txt')
    if not requirements_file.exists():
        print("✗ requirements.txt file not found!")
        return False
    
    # Show what will be installed
    print("Installing Python dependencies from requirements.txt...")
    print("This includes:")
    print("  • Flask and web framework dependencies")
    print("  • Machine learning libraries (scikit-learn, xgboost)")
    print("  • RAG dependencies (sentence-transformers, faiss, langchain)")
    print("  • Gemini AI (google-generativeai)")
    print("  • URL processing and security libraries")
    print("  • Google Safe Browsing API integration (requests)")
    
    result = subprocess.run([str(PYTHON_EXECUTABLE), '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                          capture_output=True, text=True)
    if result.returncode != 0:
        print(f"✗ Failed to install Python dependencies: {result.stderr}")
        print("Try running manually: pip install -r requirements.txt")
        return False
    print("✓ Python dependencies installed successfully")
    
    # Install frontend dependencies
    print("Installing frontend dependencies...")
    frontend_dir = Path('frontend')
    if frontend_dir.exists():
        result = subprocess.run(['npm', 'install'], cwd=frontend_dir, 
                              capture_output=True, text=True, shell=True)
        if result.returncode != 0:
            print(f"✗ Failed to install frontend dependencies: {result.stderr}")
            return False
        print("✓ Frontend dependencies installed successfully")
    else:
        print("⚠ Frontend directory not found, skipping frontend dependencies")
    
    return True

def start_backend():
    """Start the Flask backend"""
    print("Starting backend server...")
    try:
        # Use api.py as the main backend
        process = subprocess.Popen([str(PYTHON_EXECUTABLE), 'api.py'], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT,
                                 text=True,
                                 bufsize=1,
                                 universal_newlines=True,
                                 env=SUBPROCESS_ENV,
                                 cwd=PROJECT_ROOT)
        
        # Monitor the process output in real-time
        for line in iter(process.stdout.readline, ''):
            if line:
                print(f"Backend: {line.strip()}")
        
        # Wait for process to complete
        process.wait()
        
        # Check if process failed
        if process.returncode != 0:
            print(f"Backend process exited with code: {process.returncode}")
            
    except KeyboardInterrupt:
        print("\nBackend stopped")
    except Exception as e:
        print(f"Backend error: {e}")

def start_rag_backend():
    """Start the RAG backend (Chatbot service)"""
    print("Starting RAG backend (Chatbot)...")
    try:
        process = subprocess.Popen([str(PYTHON_EXECUTABLE), 'rag_backend.py'], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT,
                                 text=True,
                                 bufsize=1,
                                 universal_newlines=True,
                                 env=SUBPROCESS_ENV,
                                 cwd=PROJECT_ROOT)
        
        # Monitor the process output in real-time
        for line in iter(process.stdout.readline, ''):
            if line:
                print(f"Chatbot: {line.strip()}")
        
        # Wait for process to complete
        process.wait()
        
        # Check if process failed
        if process.returncode != 0:
            print(f"Chatbot process exited with code: {process.returncode}")
            
    except KeyboardInterrupt:
        print("\nChatbot stopped")
    except Exception as e:
        print(f"Chatbot error: {e}")

def start_frontend():
    """Start the React frontend"""
    print("Starting frontend server...")
    frontend_dir = Path('frontend')
    if frontend_dir.exists():
        try:
            subprocess.run(['npm', 'start'], cwd=frontend_dir, shell=True, env=SUBPROCESS_ENV)
        except KeyboardInterrupt:
            print("\nFrontend stopped")
        except Exception as e:
            print(f"Frontend error: {e}")
    else:
        print("Frontend directory not found")

def check_service_status():
    """Check if services are already running"""
    import requests
    
    services = {
        "Backend API": "http://localhost:5001/api/health",
        "Chatbot": "http://localhost:5002/api/rag/health"
    }
    
    running_services = []
    for service_name, url in services.items():
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                running_services.append(service_name)
        except:
            pass
    
    if running_services:
        print(f"\n⚠ Warning: These services are already running: {', '.join(running_services)}")
        choice = input("Continue anyway? (y/n): ").lower().strip()
        if choice not in ['y', 'yes']:
            return False
    
    return True

def main():
    """Main startup function"""
    print("🚀 Phishing URL Detection System")
    print("=" * 50)
    print("This will start:")
    print("  • Backend API (Phishing Detection) - Port 5001")
    print("  • Chatbot (AI Education Assistant with RAG) - Port 5002") 
    print("  • Frontend (React Web Interface) - Port 3000")
    print("=" * 50)
    print("Features:")
    print("  • AI-powered phishing URL detection")
    print("  • Google Safe Browsing API threat intelligence")
    print("  • Gemini AI chatbot for cybersecurity education")
    print("  • True RAG (Retrieval-Augmented Generation)")
    print("  • Modern React frontend")
    print("=" * 50)
    
    # Check environment variables
    check_environment_variables()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install missing dependencies.")
        return
    
    # Check if services are already running
    if not check_service_status():
        print("❌ Startup cancelled.")
        return
    
    # Ask if user wants to install dependencies
    install_deps = input("\n📦 Install/update dependencies? (y/n): ").lower().strip()
    if install_deps in ['y', 'yes']:
        if not install_dependencies():
            print("❌ Failed to install dependencies")
            return
    
    print("\n🚀 Starting all services...")
    print("📍 Backend API: http://localhost:5001")
    print("🤖 Chatbot: http://localhost:5002") 
    print("🌐 Frontend: http://localhost:3000")
    print("🛡️  Threat Intelligence: Google Safe Browsing API integrated")
    print("⏹️  Press Ctrl+C to stop all services")
    print("-" * 50)
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=start_backend, daemon=True)
    backend_thread.start()
    
    # Wait a moment for backend to start
    time.sleep(3)
    
    # Start RAG backend (Chatbot) in a separate thread
    rag_thread = threading.Thread(target=start_rag_backend, daemon=True)
    rag_thread.start()
    
    # Wait a moment for RAG backend to start
    time.sleep(3)
    
    # Start frontend
    try:
        start_frontend()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        print("✅ All services stopped")

if __name__ == "__main__":
    main()
