"""
Main entry point for Options Analytics Project

Usage:
    python main.py --dashboard    # Run Streamlit dashboard
    python main.py --test          # Run unit tests
    python main.py                 # Run dashboard (default)
"""

import sys
import subprocess
import argparse


def run_dashboard():
    """Run the Streamlit dashboard."""
    import os
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard", "app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path])


def run_tests():
    """Run unit tests."""
    import unittest
    import os

    # Add project root to path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)

    # Discover and run tests
    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Options Analytics Project")
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Run Streamlit dashboard",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run unit tests",
    )

    args = parser.parse_args()

    if args.test:
        run_tests()
    else:
        # Default: run dashboard
        run_dashboard()


if __name__ == "__main__":
    main()
