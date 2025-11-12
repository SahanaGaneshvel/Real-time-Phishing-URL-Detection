"""
Vercel serverless function entry point for Flask app
This file is required for Vercel to properly deploy the Flask application
"""

import sys
import os

# Add parent directory to path to import app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

# Vercel expects the WSGI application to be exposed as 'app' or 'application'
# The Flask app instance is already named 'app', so we can use it directly
application = app

# For Vercel, we can also export it as 'app'
__all__ = ['app', 'application']

