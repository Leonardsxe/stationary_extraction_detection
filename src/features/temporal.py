"""
Temporal feature engineering for fuel theft detection.
"""
import pandas as pd
import numpy as np
from typing import Optional
import logging

from ..config.config import Config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class TemporalFeatureEngineer:
    """Creates temporal-based features."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize TemporalFeatureEngineer.
        
        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or Config()
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all temporal features.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with temporal features
        """
        # Sort by vehicle and time
        if all(col in df.columns for col in ['Vehicle_ID', 'Tiempo']):
            df = df.sort_values(['Vehicle_ID', 'Tiempo'])
        
        # Basic time features
        df = self._create_time_features(df)
        
        # Cyclical encoding
        df = self._create_cyclical_features(df)
        
        # Time differences
        df = self._create_time_differences(df)
        
        # Fuel-related temporal features
        df = self._create_fuel_temporal_features(df)
        
        # Stationary duration
        df = self._create_stationary_duration(df)
        
        # Time since refuel
        df = self._create_time_since_refuel(df)
        
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract basic time features."""
        if 'Tiempo' not in df.columns:
            logger.warning("'Tiempo' column not found, skipping time features")
            return df
        
        df['Hour'] = df['Tiempo'].dt.hour
        df['DayOfWeek'] = df['Tiempo'].dt.dayofweek
        df['Day'] = df['Tiempo'].dt.day
        df['Month'] = df['Tiempo'].dt.month
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        
        # Night time indicator
        night_start, night_end = self.config.features.night_hours
        df['Is_NightTime'] = (
            (df['Hour'] >= night_start) | (df['Hour'] <= night_end)
        ).astype(int)
        
        # Time of day categories
        df['TimeOfDay'] = pd.cut(
            df['Hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
            include_lowest=True
        )
        
        return df
    
    def _create_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cyclical encoding for temporal features."""
        if 'Hour' in df.columns:
            df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
            df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        
        if 'DayOfWeek' in df.columns:
            df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
            df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        
        if 'Day' in df.columns:
            df['Day_Sin'] = np.sin(2 * np.pi * df['Day'] / 31)
            df['Day_Cos'] = np.cos(2 * np.pi * df['Day'] / 31)
        
        if 'Month' in df.columns:
            df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
            df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        return df
    
    def _create_time_differences(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time difference features."""
        if 'Tiempo' not in df.columns:
            return df
        
        # Time since previous record
        if 'Vehicle_ID' in df.columns:
            df['Time_Diff'] = df.groupby('Vehicle_ID')['Tiempo'].diff().dt.total_seconds() / 60  # minutes
        else:
            df['Time_Diff'] = df['Tiempo'].diff().dt.total_seconds() / 60
        
        # Categories for time differences
        df['Time_Diff_Category'] = pd.cut(
            df['Time_Diff'],
            bins=[0, 5, 15, 30, 60, np.inf],
            labels=['Very_Short', 'Short', 'Medium', 'Long', 'Very_Long']
        )
        
        return df
    
    def _create_fuel_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create fuel-related temporal features."""
        if 'Tanque Total' not in df.columns:
            return df
        
        # Fuel differences
        if 'Vehicle_ID' in df.columns:
            df['Fuel_Diff'] = df.groupby('Vehicle_ID')['Tanque Total'].diff()
        else:
            df['Fuel_Diff'] = df['Tanque Total'].diff()
        
        # Fuel consumption rate
        if 'Time_Diff' in df.columns:
            df['Fuel_Rate'] = df['Fuel_Diff'] / df['Time_Diff']
            # Handle division by zero
            df['Fuel_Rate'] = df['Fuel_Rate'].replace([np.inf, -np.inf], 0)
            df['Fuel_Rate'] = df['Fuel_Rate'].fillna(0)
        
        # Movement indicator
        if 'Velocidad' in df.columns:
            df['Is_Stationary'] = (df['Velocidad'] == 0).astype(int)
        
        return df
    
    def _create_stationary_duration(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate cumulative stationary duration."""
        if not all(col in df.columns for col in ['Is_Stationary', 'Time_Diff']):
            logger.warning("Required columns for stationary duration not found")
            return df
        
        df['Stationary_Duration'] = 0
        
        if 'Vehicle_ID' in df.columns:
            for vehicle in df['Vehicle_ID'].unique():
                mask = df['Vehicle_ID'] == vehicle
                vehicle_data = df.loc[mask]
                
                # Identify stationary groups
                stationary_groups = (
                    vehicle_data['Is_Stationary'] != vehicle_data['Is_Stationary'].shift()
                ).cumsum()
                
                # Calculate cumulative duration for stationary periods
                stationary_duration = vehicle_data.groupby(stationary_groups)['Time_Diff'].cumsum()
                stationary_duration = stationary_duration * vehicle_data['Is_Stationary']
                
                df.loc[mask, 'Stationary_Duration'] = stationary_duration
        
        return df
    
    def _create_time_since_refuel(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate time since last refuel event."""
        if 'Fuel_Diff' not in df.columns:
            return df
        
        # Identify refuel events
        df['Is_Refuel'] = (df['Fuel_Diff'] > self.config.features.refuel_threshold).astype(int)
        
        # Calculate time since refuel
        df['Time_Since_Refuel'] = 0
        
        if 'Vehicle_ID' in df.columns:
            for vehicle in df['Vehicle_ID'].unique():
                mask = df['Vehicle_ID'] == vehicle
                vehicle_data = df.loc[mask].copy()
                
                last_refuel_time = pd.NaT
                time_since_refuel = []
                
                for idx, row in vehicle_data.iterrows():
                    if pd.isna(last_refuel_time):
                        time_since_refuel.append(0)
                    else:
                        hours_diff = (row['Tiempo'] - last_refuel_time).total_seconds() / 3600
                        time_since_refuel.append(hours_diff)
                    
                    if row['Is_Refuel']:
                        last_refuel_time = row['Tiempo']
                
                df.loc[mask, 'Time_Since_Refuel'] = time_since_refuel
        
        # Create sudden drop indicator
        df['Is_Sudden_Drop'] = (
            (df['Fuel_Diff'] < -5) & 
            (df['Time_Diff'] < 30)  # Less than 30 minutes
        ).astype(int)
        
        return df