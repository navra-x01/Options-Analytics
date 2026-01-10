"""
Streamlit Dashboard for Options Analytics - Root Entry Point

This file serves as the main entry point for Streamlit Cloud deployment.
It imports and runs the dashboard application.
"""

import sys
import os

# Add current directory to path for imports
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
except Exception:
    # Fallback: try to add current working directory
    try:
        cwd = os.getcwd()
        if cwd not in sys.path:
            sys.path.insert(0, cwd)
    except Exception:
        pass

# Import and run the dashboard app
# This allows Streamlit Cloud to find the app at the root level
try:
    from dashboard.app import *
except ImportError as e:
    import streamlit as st
    st.error(f"Import Error: {e}")
    st.error("Please ensure all dependencies are installed and the project structure is correct.")
    raise
