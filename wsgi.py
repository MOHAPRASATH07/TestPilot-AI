"""
WSGI entry point for TestPilot AI Flask application.
Allows standard `flask run` and production WSGI servers (Gunicorn/Waitress) to locate the application automatically.
"""

import os
from app_flask import app

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    app.run(host=host, port=port, debug=False)
