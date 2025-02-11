# src/utils/__init__.py

from .path_manager import setup_project_root, find_project_root

# Automatically set up the project root when utils is imported
setup_project_root()
