#!/usr/bin/env python3
"""
Simple runner script for the Flask application
"""
from app import app

if __name__ == '__main__':
    print("Starting Cryptographic Algorithm Identifier Web Interface...")
    print("Access the application at: http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
