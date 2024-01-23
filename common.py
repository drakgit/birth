import os
from pathlib import Path

def get_current_directory():
    return str(os.getenv("YOLO_HOME", default=str(os.getcwd())))

def get_pva_yolo_home():
    """Get the home directory for storing weights and models.

    Returns:
        str: the home directory.
    """
    return str(os.getenv("YOLO_HOME", default=str(os.getcwd()))) + "\\runs\\segment\\train8\\weights"