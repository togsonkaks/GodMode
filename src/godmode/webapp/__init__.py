"""
Webapp package.

Important: avoid importing FastAPI app wiring at package import time to prevent circular imports
with modules that are used outside the web server (e.g., report generation).
"""

__all__: list[str] = []


