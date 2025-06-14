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


def get_date_range(df: pd.DataFrame, date_column: str = 'timestamp'
                   ) -> Dict[str, Any]:
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

def _convert_for_json(obj: Any, max_size: int = 1000) -> Any:
    """
    Recursively convert an object to be JSON-serializable with size limits.
    
    Args:
        obj: Object to convert
        max_size: Maximum size for arrays/dataframes before summarizing
        
    Returns:
        JSON-serializable object
    """
    if isinstance(obj, dict):
        # Convert dict keys to strings and recursively process values
        return {str(k): _convert_for_json(v, max_size) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # Truncate long lists
        if len(obj) > max_size:
            return {
                '_type': 'truncated_list',
                'length': len(obj),
                'sample': [_convert_for_json(elem, max_size) for elem in obj[:10]],
                'message': f'List truncated - showing first 10 of {len(obj)} items'
            }
        return [_convert_for_json(elem, max_size) for elem in obj]
    elif isinstance(obj, np.ndarray):
        # Summarize large arrays
        if obj.size > max_size:
            return {
                '_type': 'numpy_array_summary',
                'shape': list(obj.shape),
                'dtype': str(obj.dtype),
                'size': int(obj.size),
                'sample': obj.flatten()[:10].tolist(),
                'stats': {
                    'mean': float(obj.mean()) if obj.dtype.kind in 'biufc' else None,
                    'std': float(obj.std()) if obj.dtype.kind in 'biufc' else None,
                    'min': float(obj.min()) if obj.dtype.kind in 'biufc' else None,
                    'max': float(obj.max()) if obj.dtype.kind in 'biufc' else None,
                }
            }
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        # Summarize large Series
        if len(obj) > max_size:
            return {
                '_type': 'pandas_series_summary',
                'length': len(obj),
                'dtype': str(obj.dtype),
                'name': obj.name,
                'unique_values': int(obj.nunique()),
                'sample': obj.head(10).to_dict(),
                'stats': obj.describe().to_dict() if obj.dtype.kind in 'biufc' else None
            }
        return obj.to_dict()
    elif isinstance(obj, pd.DataFrame):
        # Summarize DataFrames
        if len(obj) > max_size or obj.shape[1] > 50:
            return {
                '_type': 'pandas_dataframe_summary',
                'shape': list(obj.shape),
                'columns': list(obj.columns),
                'dtypes': obj.dtypes.astype(str).to_dict(),
                'memory_usage_mb': obj.memory_usage(deep=True).sum() / 1024**2,
                'sample_rows': obj.head(10).to_dict('records'),
                'numeric_stats': obj.describe().to_dict() if any(obj.dtypes.apply(lambda x: x.kind in 'biufc')) else None
            }
        return obj.to_dict(orient='records')
    elif isinstance(obj, (np.integer, np.floating)):
        # Convert numpy types to Python types
        return obj.item()
    elif hasattr(obj, '__dict__'):
        # Don't convert model objects - just return their class name
        if 'sklearn' in str(type(obj)) or 'Model' in str(type(obj)):
            return {
                '_type': 'model_object',
                'class': str(type(obj)),
                'module': str(type(obj).__module__)
            }
        # Convert other objects with __dict__ to dict
        return {str(k): _convert_for_json(v, max_size) for k, v in obj.__dict__.items()}
    else:
        # Return as-is for JSON-serializable types
        return obj


def create_summary_dict(data: Any, max_items: int = 10) -> Any:
    """
    Create a summary version of data for JSON storage.
    
    Args:
        data: Data to summarize
        max_items: Maximum number of items to include in lists
        
    Returns:
        Summarized version of data
    """
    if isinstance(data, pd.DataFrame):
        return {
            'shape': list(data.shape),
            'columns': list(data.columns),
            'dtypes': data.dtypes.astype(str).to_dict(),
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024**2,
            'sample_rows': data.head(max_items).to_dict('records') if len(data) > 0 else []
        }
    elif isinstance(data, pd.Series):
        return {
            'length': len(data),
            'dtype': str(data.dtype),
            'unique_values': int(data.nunique()) if not data.empty else 0,
            'sample_values': data.head(max_items).tolist() if len(data) > 0 else []
        }
    elif isinstance(data, np.ndarray):
        return {
            'shape': list(data.shape),
            'dtype': str(data.dtype),
            'size': data.size,
            'sample_values': data.flatten()[:max_items].tolist() if data.size > 0 else []
        }
    elif isinstance(data, dict):
        return {k: create_summary_dict(v, max_items) for k, v in data.items()}
    elif isinstance(data, list):
        if len(data) > max_items:
            return {
                'length': len(data),
                'sample': [create_summary_dict(item, max_items) for item in data[:max_items]],
                'truncated': True
            }
        return [create_summary_dict(item, max_items) for item in data]
    else:
        return data


def summarize_model_results(results: Dict[str, Any], model_type: str) -> Dict[str, Any]:
    """
    Create summary of model results for JSON storage.
    
    Args:
        results: Model results dictionary
        model_type: Type of model ('unsupervised', 'supervised', 'ensemble')
        
    Returns:
        Summarized results
    """
    if model_type == 'unsupervised':
        summary = {
            'models_used': list(results.get('models', {}).keys()) if 'models' in results else [],
            'n_anomalies_detected': {},
            'execution_time': results.get('execution_time', 0)
        }
        
        # Summarize predictions
        if 'predictions' in results:
            for model, preds in results['predictions'].items():
                if isinstance(preds, (pd.Series, np.ndarray)):
                    n_anomalies = int((preds == -1).sum()) if hasattr(preds, 'sum') else 0
                    summary['n_anomalies_detected'][model] = n_anomalies
        
        # Include metrics if available
        if 'metrics' in results:
            summary['metrics'] = results['metrics']
            
    elif model_type == 'supervised':
        summary = {
            'models_trained': list(results.get('models', {}).keys()) if 'models' in results else [],
            'best_model': results.get('best_model', 'unknown'),
            'training_set_size': None,
            'test_set_size': None,
            'execution_time': results.get('execution_time', 0)
        }
        
        # Add data split information
        if 'X_train' in results:
            summary['training_set_size'] = len(results['X_train'])
        if 'X_test' in results:
            summary['test_set_size'] = len(results['X_test'])
        
        # Add model metrics
        if 'metrics' in results:
            summary['metrics'] = {}
            for model, metrics in results['metrics'].items():
                if isinstance(metrics, dict):
                    # Only keep scalar metrics
                    summary['metrics'][model] = {
                        k: v for k, v in metrics.items() 
                        if isinstance(v, (int, float, str, bool))
                    }
        
        # Add feature importance summary
        if 'feature_importance' in results:
            summary['top_10_features'] = {}
            for model, importance in results['feature_importance'].items():
                if isinstance(importance, pd.DataFrame):
                    top_features = importance.nlargest(10, 'importance')[['feature', 'importance']]
                    summary['top_10_features'][model] = top_features.to_dict('records')
                    
    elif model_type == 'ensemble':
        summary = {
            'execution_time': results.get('execution_time', 0)
        }
        
        # Add ensemble-specific metrics
        if 'threshold' in results:
            summary['threshold'] = float(results['threshold'])
        if 'weights' in results:
            summary['weights'] = results['weights']
    
    return summary