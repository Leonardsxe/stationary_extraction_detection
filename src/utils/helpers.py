"""
General utility functions for the fuel theft detection project.
"""
import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import hashlib
from datetime import datetime


def save_dataframe(df: pd.DataFrame, filepath: Union[str, Path], **kwargs) -> Path:
    """
    Save DataFrame to file with automatic format detection.
    
    Args:
        df: DataFrame to save
        filepath: Output file path
        **kwargs: Additional arguments passed to pandas save function
        
    Returns:
        Path to saved file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if filepath.suffix == '.csv':
        df.to_csv(filepath, index=False, **kwargs)
    elif filepath.suffix == '.parquet':
        df.to_parquet(filepath, index=False, **kwargs)
    elif filepath.suffix == '.xlsx':
        df.to_excel(filepath, index=False, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    return filepath


def load_dataframe(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Load DataFrame from file with automatic format detection.
    
    Args:
        filepath: Input file path
        **kwargs: Additional arguments passed to pandas load function
        
    Returns:
        Loaded DataFrame
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if filepath.suffix == '.csv':
        return pd.read_csv(filepath, **kwargs)
    elif filepath.suffix == '.parquet':
        return pd.read_parquet(filepath, **kwargs)
    elif filepath.suffix == '.xlsx':
        return pd.read_excel(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def save_model(model: Any, filepath: Union[str, Path]) -> Path:
    """
    Save model to pickle file.
    
    Args:
        model: Model object to save
        filepath: Output file path
        
    Returns:
        Path to saved file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    return filepath


def load_model(filepath: Union[str, Path]) -> Any:
    """
    Load model from pickle file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded model object
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_json(data: Dict, filepath: Union[str, Path], indent: int = 2) -> Path:
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Output file path
        indent: JSON indentation
        
    Returns:
        Path to saved file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert all keys to strings for JSON compatibility
    data_to_save = _convert_for_json(data)
    
    with open(filepath, 'w') as f:
        json.dump(data_to_save, f, indent=indent, default=str)
    
    return filepath


def _convert_for_json(obj: Any) -> Any:
    """
    Recursively convert an object to be JSON-serializable.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable object
    """
    if isinstance(obj, dict):
        # Convert dict keys to strings and recursively process values
        return {str(k): _convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # Recursively process list/tuple elements
        return [_convert_for_json(elem) for elem in obj]
    elif isinstance(obj, np.ndarray):
        # Convert numpy arrays to lists
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        # Convert pandas Series to dict
        return obj.to_dict()
    elif isinstance(obj, pd.DataFrame):
        # Convert pandas DataFrame to dict
        return obj.to_dict(orient='records')
    elif isinstance(obj, (np.integer, np.floating)):
        # Convert numpy types to Python types
        return obj.item()
    elif hasattr(obj, '__dict__'):
        # Convert objects with __dict__ to dict
        return {str(k): _convert_for_json(v) for k, v in obj.__dict__.items()}
    else:
        # Return as-is for JSON-serializable types
        return obj


def load_json(filepath: Union[str, Path]) -> Dict:
    """
    Load dictionary from JSON file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded dictionary
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_memory_usage(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate memory usage of DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with memory usage statistics
    """
    memory_usage = df.memory_usage(deep=True)
    
    return {
        'total_mb': memory_usage.sum() / 1024**2,
        'average_per_column_mb': memory_usage.mean() / 1024**2,
        'per_column_mb': (memory_usage / 1024**2).to_dict()
    }


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Reduce memory usage of DataFrame by optimizing data types.
    
    Args:
        df: Input DataFrame
        verbose: Whether to print optimization details
        
    Returns:
        Optimized DataFrame
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != 'object':
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    if verbose:
        print(f'Memory usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB '
              f'({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    
    return df


def create_experiment_id(params: Dict[str, Any]) -> str:
    """
    Create unique experiment ID from parameters.
    
    Args:
        params: Dictionary of parameters
        
    Returns:
        Unique experiment ID
    """
    # Sort parameters for consistency
    sorted_params = json.dumps(params, sort_keys=True)
    
    # Create hash
    hash_object = hashlib.md5(sorted_params.encode())
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return f"{timestamp}_{hash_object.hexdigest()[:8]}"


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    Flatten nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key for recursion
        sep: Separator between keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_date_range(df: pd.DataFrame, date_column: str = 'Tiempo') -> Dict[str, Any]:
    """
    Get date range information from DataFrame.
    
    Args:
        df: Input DataFrame
        date_column: Name of date column
        
    Returns:
        Dictionary with date range information
    """
    if date_column not in df.columns:
        return {}
    
    dates = pd.to_datetime(df[date_column])
    
    return {
        'start_date': dates.min(),
        'end_date': dates.max(),
        'duration_days': (dates.max() - dates.min()).days,
        'unique_dates': dates.dt.date.nunique()
    }


def check_duplicate_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Check for duplicate columns in DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with duplicate column information
    """
    duplicates = df.columns[df.columns.duplicated()].tolist()
    
    # Group duplicates
    duplicate_groups = {}
    for col in set(duplicates):
        duplicate_groups[col] = [i for i, x in enumerate(df.columns) if x == col]
    
    return {
        'has_duplicates': len(duplicates) > 0,
        'n_duplicates': len(duplicates),
        'duplicate_columns': list(set(duplicates)),
        'duplicate_groups': duplicate_groups
    }


def remove_duplicate_columns(df: pd.DataFrame, keep: str = 'first') -> pd.DataFrame:
    """
    Remove duplicate columns from DataFrame.
    
    Args:
        df: Input DataFrame
        keep: Which duplicate to keep ('first', 'last')
        
    Returns:
        DataFrame without duplicate columns
    """
    return df.loc[:, ~df.columns.duplicated(keep=keep)]