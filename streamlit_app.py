"""
Streamlit Dashboard for Options Analytics - Root Entry Point

This file serves as the main entry point for Streamlit Cloud deployment.
It imports and runs the dashboard application.
"""

import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import and run the dashboard app
# This allows Streamlit Cloud to find the app at the root level
from dashboard.app import *
