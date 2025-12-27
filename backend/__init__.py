# =============================================================================
# Backend Package
# =============================================================================
"""
Acme CRM AI Companion - Backend API Package

This package provides the FastAPI backend for the CRM AI Companion.

Modules:
    - main: Application entry point and factory
    - config: Configuration management
    - api: Modular API route definitions
    - middleware: Request/response middleware
    - exceptions: Custom exceptions and error handling

Usage:
    # Run with uvicorn
    uvicorn backend.main:app --reload
    
    # Or run directly
    python -m backend.main
"""

from backend.config import get_settings, Settings
from backend.exceptions import APIError, ValidationError, NotFoundError, AgentError

__all__ = [
    "get_settings",
    "Settings",
    "APIError",
    "ValidationError",
    "NotFoundError",
    "AgentError",
]
