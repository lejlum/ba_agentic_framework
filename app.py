"""Entry point for Hugging Face Spaces deployment.
Imports the Flask server object so gunicorn can find it via 'app:server'.
"""
import sys
import os

# Add the repo root to sys.path so the package can be imported regardless
# of the working directory gunicorn uses when starting the app.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from waste_recycling_chatbot_pa2.dashboard.dashboard_app import server