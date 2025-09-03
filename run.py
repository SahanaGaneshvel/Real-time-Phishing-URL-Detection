#!/usr/bin/env python3
"""
Startup script for the Phishing URL Detection application
"""

import os
import sys
import subprocess
import time

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import flask
        import sklearn
        import pandas
        import numpy
        print("✓ All required dependencies are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_model():
    """Check if the trained model exists"""
    model_path = os.path.join('model', 'phishing_model.pkl')
    if os.path.exists(model_path):
        print("✓ Trained model found")
        return True
    else:
        print("✗ Trained model not found")
        return False

def train_model():
    """Train the model if it doesn't exist"""
    print("\nTraining model...")
    try:
        result = subprocess.run([sys.executable, 'train_model.py'], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✓ Model training completed successfully")
            return True
        else:
            print(f"✗ Model training failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ Model training timed out")
        return False
    except Exception as e:
        print(f"✗ Model training error: {e}")
        return False

def start_application():
    """Start the Flask application"""
    print("\nStarting Flask application...")
    print("The application will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the application")
    print("-" * 50)
    
    try:
        # Start the Flask app
        subprocess.run([sys.executable, 'app.py'])
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"Error starting application: {e}")

def main():
    """Main startup function"""
    print("Phishing URL Detection System")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check if model exists, train if needed
    if not check_model():
        print("\nNo trained model found. Training a new model...")
        if not train_model():
            print("Failed to train model. Exiting.")
            return
    
    # Start the application
    start_application()

if __name__ == "__main__":
    main() 