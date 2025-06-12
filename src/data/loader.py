"""
Data loading utilities for fuel theft detection.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
import logging
from dataclasses import dataclass

from ..config.config import Config
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DataInfo:
    """Container for data loading results."""
    data: pd.DataFrame
    metadata: Dict[str, Any]
    errors: List[str]


class DataLoader:
    """Handles loading and initial processing of fuel data."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize DataLoader.
        
        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or Config()
        self.data = None
        self.metadata = {}
        
    def load_excel_files(self, file_paths: List[Union[str, Path]]) -> pd.DataFrame:
        """
        Load and combine multiple Excel files.
        
        Args:
            file_paths: List of paths to Excel files
            
        Returns:
            Combined DataFrame
            
        Raises:
            ValueError: If no valid files are loaded
        """
        logger.info(f"Loading {len(file_paths)} Excel files...")
        
        all_data = []
        errors = []
        
        for file_path in file_paths:
            try:
                df = self._load_single_file(file_path)
                if df is not None:
                    all_data.append(df)
                    logger.info(f"Successfully loaded {file_path}: {len(df)} records")
            except Exception as e:
                error_msg = f"Error loading {file_path}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        if not all_data:
            raise ValueError("No valid data files were loaded")
        
        # Combine all dataframes
        combined_df = pd.concat(all_data, ignore_index=True, sort=False)
        
        # Store metadata
        self.metadata = {
            'total_files': len(file_paths),
            'loaded_files': len(all_data),
            'total_records': len(combined_df),
            'vehicles': combined_df['Vehicle_ID'].nunique() if 'Vehicle_ID' in combined_df else 0,
            'errors': errors
        }
        
        logger.info(f"Combined data: {self.metadata['total_records']} records from "
                   f"{self.metadata['loaded_files']} files")
        
        self.data = combined_df
        return combined_df
    
    def _load_single_file(self, file_path: Union[str, Path]) -> Optional[pd.DataFrame]:
        """
        Load a single Excel file with error handling.
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            DataFrame or None if error
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return None
        
        try:
            # Extract vehicle ID from filename
            vehicle_id = file_path.stem.split('_')[0]
            
            # Read Excel file
            df = pd.read_excel(file_path)
            
            # Add metadata columns
            df['Vehicle_ID'] = vehicle_id
            df['Source_File'] = str(file_path)
            
            # Standardize column names
            df.columns = df.columns.str.strip().str.replace('\n', ' ')
            
            # Apply column mappings
            df.rename(columns=self.config.data.column_mappings, inplace=True)

            if hasattr(self.config.data, 'missing_indicators') and self.config.data.missing_indicators:
                logger.debug(
                    f"In {file_path}: Applying replacement of missing_indicators: {self.config.data.missing_indicators}"
                )
                for indicator in self.config.data.missing_indicators:
                    if pd.isna(indicator): # Avoid issues if NaN is somehow in indicators
                        continue
                    
                    mask = df == indicator
                    df = df.mask(mask, np.nan)
                    # df = df.replace(indicator, np.nan) # Reassign df to apply changes

            # Initial datetime conversion
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                
                # Remove rows with invalid dates
                invalid_dates_mask = df['timestamp'].isna()
                num_invalid_dates = invalid_dates_mask.sum()
                if num_invalid_dates > 0:
                    logger.warning(f"In {file_path}, 'timestamp' column: Removing {num_invalid_dates} rows with invalid dates.")
                    df = df.dropna(subset=['timestamp'])
            
            # Convert designated numeric columns to numeric, coercing errors.
            # This handles non-numeric strings (e.g., '-----') by converting them to NaN,
            # making them Parquet-compatible.
            # if hasattr(self.config.data, 'numeric_columns') and self.config.data.numeric_columns:
            #     for col_name in self.config.data.numeric_columns:
            #         if col_name in df.columns:
            #             # Process only if it's an object type, as pd.to_numeric is idempotent on numeric types
            #             if df[col_name].dtype == 'object':
            #                 # Identify values that are not NaN but will become NaN after coercion
            #                 # This helps in logging only newly coerced NaNs from strings.
            #                 problematic_values_mask = df[col_name].notna() & pd.to_numeric(df[col_name], errors='coerce').isna()
            #                 num_coerced = problematic_values_mask.sum()
            #                 if num_coerced > 0:
            #                     logger.warning(
            #                         f"In {file_path}, '{col_name}' column: "
            #                         f"{num_coerced} non-numeric values (e.g., '-----') converted to NaN."
            #                     )
            #                 df[col_name] = pd.to_numeric(df[col_name], errors='coerce')


            return df
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return None
    
    def load_from_directory(self, directory: Optional[Union[str, Path]] = None, 
                          pattern: str = "*.xlsx") -> pd.DataFrame:
        """
        Load all Excel files from a directory.
        
        Args:
            directory: Directory path. If None, uses config default
            pattern: File pattern to match
            
        Returns:
            Combined DataFrame
        """
        directory = Path(directory) if directory else self.config.data.raw_data_path
        
        if not directory.exists():
            raise ValueError(f"Directory not found: {directory}")
        
        file_paths = list(directory.glob(pattern))
        
        if not file_paths:
            raise ValueError(f"No files matching pattern '{pattern}' found in {directory}")
        
        logger.info(f"Found {len(file_paths)} files matching pattern '{pattern}'")
        
        return self.load_excel_files(file_paths)
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = "processed_data.csv",
                          directory: Optional[Union[str, Path]] = None) -> Path:
        """
        Save processed data to CSV.
        
        Args:
            df: DataFrame to save
            filename: Output filename
            directory: Output directory. If None, uses config default
            
        Returns:
            Path to saved file
        """
        directory = Path(directory) if directory else self.config.data.processed_data_path
        directory.mkdir(parents=True, exist_ok=True)
        
        output_path = directory / filename
        df.to_csv(output_path, index=False)
        
        logger.info(f"Saved processed data to {output_path}")
        return output_path
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about loaded data."""
        if self.data is None:
            return {"error": "No data loaded"}
        
        info = {
            **self.metadata,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024**2,
            'date_range': {
                'start': self.data['timestamp'].min() if 'timestamp' in self.data else None,
                'end': self.data['timestamp'].max() if 'timestamp' in self.data else None
            }
        }
        
        return info