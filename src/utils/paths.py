"""
Path management utilities
"""

from pathlib import Path
from typing import Union


def get_project_root() -> Path:
    """Get project root directory"""
    # Assuming this file is in src/utils, go up 2 levels
    return Path(__file__).parent.parent.parent


def get_data_path(layer: str = None) -> Path:
    """
    Get data directory path
    
    Parameters:
    -----------
    layer : str, optional
        Data layer: 'raw', 'bronze', 'silver', 'gold'
    
    Returns:
    --------
    Path: Data directory path
    """
    root = get_project_root()
    if layer:
        return root / 'data' / layer
    return root / 'data'


def get_models_path(subdir: str = None) -> Path:
    """
    Get models directory path
    
    Parameters:
    -----------
    subdir : str, optional
        Subdirectory: 'checkpoints', 'artifacts', 'metadata'
    
    Returns:
    --------
    Path: Models directory path
    """
    root = get_project_root()
    if subdir:
        return root / 'models' / subdir
    return root / 'models'


def get_outputs_path(subdir: str = None) -> Path:
    """
    Get outputs directory path
    
    Parameters:
    -----------
    subdir : str, optional
        Subdirectory: 'reports', 'visualizations', 'predictions'
    
    Returns:
    --------
    Path: Outputs directory path
    """
    root = get_project_root()
    if subdir:
        return root / 'outputs' / subdir
    return root / 'outputs'

