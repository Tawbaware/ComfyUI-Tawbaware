"""Top-level package for comfyui_tawbaware."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """Tawbaware"""
__email__ = "info@tawbaware.com"
__version__ = "0.0.1"

from .src.comfyui_tawbaware.nodes import NODE_CLASS_MAPPINGS
from .src.comfyui_tawbaware.nodes import NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"
