"""
Data cleaning utilities for fuel theft detection.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

from ..config.config import Config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataCleaner:
    """Handles data cleaning and preprocessing."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize DataCleaner.
        
        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or Config()
        self.cleaning_report = {}
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all cleaning steps to the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Cleaned dataframe
        """
        logger.info("Starting data cleaning process...")
        
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Track original size
        original_rows = len(df_clean)
        
        # Apply cleaning steps
        df_clean = self._handle_missing_indicators(df_clean)
        df_clean = self._apply_ignition_rule(df_clean)
        df_clean = self._clean_datetime_columns(df_clean)
        df_clean = self._clean_numeric_columns(df_clean)
        df_clean = self._extract_coordinates(df_clean)
        df_clean = self._remove_duplicates(df_clean)
        df_clean = self._handle_outliers(df_clean)
        
        # Create cleaning report
        self.cleaning_report = {
            'original_rows': original_rows,
            'final_rows': len(df_clean),
            'rows_removed': original_rows - len(df_clean),
            'removal_percentage': (original_rows - len(df_clean)) / original_rows * 100
        }
        
        logger.info(f"Cleaning complete. Removed {self.cleaning_report['rows_removed']} rows "
                   f"({self.cleaning_report['removal_percentage']:.2f}%)")
        
        return df_clean

    def _apply_ignition_rule(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply specific cleaning rule for 'ignition_raw' based on 'speed_raw'.
        Rule: If 'ignition_raw' is NaN:
              - Set to 0 if 'speed_raw' is NaN or 'speed_raw' == 0.
              - Set to 1 if 'speed_raw' is not NaN and 'speed_raw' != 0.
        """
        logger.debug("Applying 'ignition_raw' cleaning rule based on 'speed_raw'...")
        
        ignition_col = 'ignition_raw'
        speed_col = 'speed_raw'

        if ignition_col not in df.columns:
            logger.warning(f"'{ignition_col}' column not found. Skipping ignition cleaning rule.")
            return df
        
        if speed_col not in df.columns:
            logger.warning(
                f"'{speed_col}' column not found. "
                f"Skipping '{ignition_col}' cleaning rule as it depends on '{speed_col}'."
            )
            return df

        ignition_nan_mask = df[ignition_col].isna()
        num_ignition_nan_before = ignition_nan_mask.sum()

        if num_ignition_nan_before == 0:
            logger.debug(f"No NaN values found in '{ignition_col}' to apply rule.")
            return df

        # Condition: 'speed_raw' is NaN or 'speed_raw' == 0
        condition_set_ignition_to_0 = df[speed_col].isna() | (df[speed_col] == 0)
        df.loc[ignition_nan_mask & condition_set_ignition_to_0, ignition_col] = 0
        
        # Condition: 'speed_raw' is not NaN AND 'speed_raw' != 0
        condition_set_ignition_to_1 = df[speed_col].notna() & (df[speed_col] != 0)
        df.loc[ignition_nan_mask & condition_set_ignition_to_1, ignition_col] = 1
        
        filled_count = num_ignition_nan_before - df[ignition_col].isna().sum()
        if filled_count > 0:
            logger.info(f"Applied 'ignition_raw' rule: {filled_count} NaN values in '{ignition_col}' filled based on '{speed_col}'.")
        return df
    
    def _handle_missing_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace missing value indicators with NaN."""
        logger.debug("Handling missing value indicators...")
        
        for indicator in self.config.data.missing_indicators:
            df = df.replace(indicator, np.nan)
        
        # Log missing value statistics
        missing_stats = df.isnull().sum()
        if missing_stats.any():
            logger.info(f"Missing values per column:\n{missing_stats[missing_stats > 0]}")
        
        return df
    
    def _clean_datetime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate datetime columns."""
        logger.debug("Cleaning datetime columns...")
        
        datetime_columns = ['timestamp']
        
        for col in datetime_columns:
            if col in df.columns:
                # Convert to datetime
                df[col] = pd.to_datetime(df[col], errors='coerce')
                
                # Remove rows with invalid dates
                invalid_dates = df[col].isna()
                if invalid_dates.any():
                    logger.warning(f"Removing {invalid_dates.sum()} rows with invalid {col}")
                    df = df[~invalid_dates]
        
        return df
    
    def _clean_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean numeric columns and handle non-numeric values."""
        logger.debug("Cleaning numeric columns...")
        
        for col in self.config.data.numeric_columns:
            if col in df.columns:
                # Convert to numeric, coercing errors
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Log conversion issues
                na_count = df[col].isna().sum()
                if na_count > 0:
                    logger.debug(f"Column '{col}': {na_count} non-numeric values converted to NaN")
        
        return df
    
    def _extract_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract latitude and longitude from coordinates column."""
        logger.debug("Extracting coordinates...")
        
        if 'coordinates' in df.columns:
            try:
                # Split coordinates
                coord_split = df['coordinates'].str.split(',', expand=True)
                df['Latitude'] = pd.to_numeric(coord_split[0], errors='coerce')
                df['Longitude'] = pd.to_numeric(coord_split[1], errors='coerce')
                
                # Validate coordinate ranges
                df.loc[~df['Latitude'].between(-90, 90), 'Latitude'] = np.nan
                df.loc[~df['Longitude'].between(-180, 180), 'Longitude'] = np.nan
                
                logger.info(f"Extracted coordinates: {df[['Latitude', 'Longitude']].notna().sum().sum()} valid points")
                
            except Exception as e:
                logger.error(f"Error extracting coordinates: {str(e)}")
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate records."""
        logger.debug("Removing duplicates...")
        
        # Define columns that identify a unique record
        subset_columns = ['Vehicle_ID', 'timestamp', 'total_fuel']
        subset_columns = [col for col in subset_columns if col in df.columns]
        
        if subset_columns:
            duplicates = df.duplicated(subset=subset_columns, keep='first')
            if duplicates.any():
                logger.info(f"Removing {duplicates.sum()} duplicate records")
                df = df[~duplicates]
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle extreme outliers in numeric columns."""
        logger.debug("Handling outliers...")
        
        # Define outlier thresholds
        outlier_rules = {
            'speed_raw': (0, 200),  # km/h
            'total_fuel': (0, 400),  # Gallons
        }
        
        for col, (min_val, max_val) in outlier_rules.items():
            if col in df.columns:
                outliers = ~df[col].between(min_val, max_val)
                if outliers.any():
                    logger.warning(f"Column '{col}': {outliers.sum()} outliers outside range [{min_val}, {max_val}]")
                    # Option 1: Remove outliers
                    df = df[~outliers]
                    # Option 2: Cap outliers
                    # df.loc[df[col] < min_val, col] = min_val
                    # df.loc[df[col] > max_val, col] = max_val
        
        return df
    
    def interpolate_missing_values(self, df: pd.DataFrame, 
                                 columns: Optional[List[str]] = None,
                                 method: str = 'linear') -> pd.DataFrame:
        """
        Interpolate missing values in specified columns.
        
        Args:
            df: Input dataframe
            columns: Columns to interpolate. If None, uses numeric columns
            method: Interpolation method
            
        Returns:
            DataFrame with interpolated values
        """
        logger.info("Interpolating missing values...")
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Sort by vehicle and time for proper interpolation
        if all(col in df.columns for col in ['Vehicle_ID', 'timestamp']):
            df = df.sort_values(['Vehicle_ID', 'timestamp'])
        
        for col in columns:
            if col in df.columns:
                before_na = df[col].isna().sum()
                
                # Interpolate within each vehicle group
                if 'Vehicle_ID' in df.columns:
                    df[col] = df.groupby('Vehicle_ID')[col].transform(
                        lambda x: x.interpolate(method=method, limit_direction='both')
                    )
                else:
                    df[col] = df[col].interpolate(method=method, limit_direction='both')
                
                after_na = df[col].isna().sum()
                
                if before_na > after_na:
                    logger.debug(f"Column '{col}': interpolated {before_na - after_na} values")
        
        return df
    
    def get_cleaning_report(self) -> Dict[str, any]:
        """Get the cleaning report from the last cleaning operation."""
        return self.cleaning_report
    
    def validate_cleaned_data(self, df: pd.DataFrame) -> List[str]:
        """
        Validate cleaned data and return list of issues.
        
        Args:
            df: Cleaned dataframe
            
        Returns:
            List of validation issues
        """
        issues = []
        
        # Check for required columns
        required_columns = ['Vehicle_ID', 'timestamp', 'total_fuel']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
        
        # Check for empty dataframe
        if len(df) == 0:
            issues.append("DataFrame is empty after cleaning")
        
        # Check for all-null columns
        null_columns = df.columns[df.isnull().all()].tolist()
        if null_columns:
            issues.append(f"Columns with all null values: {null_columns}")
        
        # Check datetime consistency
        if 'timestamp' in df.columns:
            if df['timestamp'].isna().any():
                issues.append("Null values found in Tiempo column")
            else:
                # Check for time travel (records going backwards in time)
                time_diffs = df.groupby('Vehicle_ID')['timestamp'].diff()
                if (time_diffs < pd.Timedelta(0)).any():
                    issues.append("Negative time differences detected (time travel)")
        
        return issues