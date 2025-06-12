"""
Behavioral feature engineering for fuel theft detection.
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict
import logging

from ..config.config import Config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class BehavioralFeatureEngineer:
    """Creates behavioral-based features."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize BehavioralFeatureEngineer.
        
        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or Config()
        self.driver_stats = None
        self.location_stats = None
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all behavioral features.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with behavioral features
        """
        # Driver-based features
        df = self._create_driver_features(df)
        
        # Location-based features
        df = self._create_location_features(df)
        
        # Vehicle-based features
        df = self._create_vehicle_features(df)
        
        # Unusual behavior indicators
        df = self._create_unusual_behavior_indicators(df)
        
        return df
    
    def _create_driver_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create driver-specific behavioral features."""
        if 'driver_name' not in df.columns:
            logger.warning("'driver_name' column not found, skipping driver features")
            return df
        
        # Calculate driver statistics
        driver_agg_features = {
            'Fuel_Diff': ['mean', 'std', 'min', 'max', lambda x: x.quantile(0.25)],
            'speed_raw': ['mean', 'std', 'max'],
            'Stationary_Duration': ['mean', 'max'],
            'Is_NightTime': ['mean'],
            'Is_Stationary': ['mean']
        }
        
        # Filter to existing columns
        driver_agg_features = {
            col: aggs for col, aggs in driver_agg_features.items() 
            if col in df.columns
        }
        
        if driver_agg_features:
            self.driver_stats = df.groupby('driver_name').agg(driver_agg_features)
            
            # Flatten column names
            self.driver_stats.columns = [
                f'Driver_{col}_{agg if isinstance(agg, str) else "q25"}' 
                for col, agg in self.driver_stats.columns
            ]
            
            # Check if columns already exist before merging
            existing_driver_cols = [col for col in self.driver_stats.columns if col in df.columns]
            if existing_driver_cols:
                logger.warning(f"Found {len(existing_driver_cols)} existing driver columns. Skipping: {existing_driver_cols}")
                # Remove existing columns from driver_stats
                self.driver_stats = self.driver_stats.drop(columns=existing_driver_cols)
            
            # Merge back to main dataframe
            if not self.driver_stats.empty:
                df = df.merge(
                    self.driver_stats,
                    left_on='driver_name',
                    right_index=True,
                    how='left',
                    suffixes=('', '_dup')  # Add suffix to handle any remaining duplicates
                )
                
                # Remove any columns with _dup suffix
                dup_cols = [col for col in df.columns if col.endswith('_dup')]
                if dup_cols:
                    logger.warning(f"Removing {len(dup_cols)} duplicate columns created during merge")
                    df = df.drop(columns=dup_cols)
            
            # Calculate deviations from driver baseline
            if 'Fuel_Diff' in df.columns and 'Driver_Fuel_Diff_mean' in df.columns:
                df['Fuel_Deviation_From_Driver'] = (
                    df['Fuel_Diff'] - df['Driver_Fuel_Diff_mean']
                ) / (df['Driver_Fuel_Diff_std'] + 1e-6)
            
            if 'speed_raw' in df.columns and 'Driver_Velocidad_mean' in df.columns:
                df['Speed_Deviation_From_Driver'] = (
                    df['speed_raw'] - df['Driver_Velocidad_mean']
                ) / (df['Driver_Velocidad_std'] + 1e-6)
        
        return df
    
    def _create_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create location-based behavioral features."""
        if 'location' not in df.columns:
            logger.warning("'location' column not found, skipping location features")
            return df
        
        # Calculate location frequency
        location_freq = df['location'].value_counts()
        total_locations = len(df)
        
        # Create location risk score (inverse of frequency)
        location_risk = 1 - (location_freq / location_freq.max())
        df['Location_Risk_Score'] = df['location'].map(location_risk).fillna(1)
        
        # Location frequency percentage
        location_freq_pct = location_freq / total_locations
        df['Location_Frequency'] = df['location'].map(location_freq_pct).fillna(0)
        
        # Is rare location (bottom 10%)
        rare_threshold = location_freq.quantile(0.1)
        rare_locations = location_freq[location_freq <= rare_threshold].index
        df['Is_Rare_Location'] = df['location'].isin(rare_locations).astype(int)
        
        # Location-based statistics
        if any(col in df.columns for col in ['Fuel_Diff', 'Is_Stationary']):
            location_stats = df.groupby('location').agg({
                col: ['mean', 'std'] for col in ['Fuel_Diff', 'Is_Stationary'] 
                if col in df.columns
            })
            
            # Store for later use
            self.location_stats = location_stats
        
        return df
    
    def _create_vehicle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create vehicle-specific behavioral features."""
        if 'Vehicle_ID' not in df.columns:
            logger.warning("'Vehicle_ID' column not found, skipping vehicle features")
            return df
        
        # Vehicle usage patterns
        vehicle_agg_features = {
            'total_fuel': ['mean', 'std'],
            'Fuel_Diff': ['mean', 'std'],
            'Is_NightTime': ['mean'],
            'Is_Stationary': ['mean']
        }
        
        # Filter to existing columns
        vehicle_agg_features = {
            col: aggs for col, aggs in vehicle_agg_features.items() 
            if col in df.columns
        }
        
        if vehicle_agg_features:
            vehicle_stats = df.groupby('Vehicle_ID').agg(vehicle_agg_features)
            
            # Flatten column names
            vehicle_stats.columns = [
                f'Vehicle_{col}_{agg}' 
                for col, agg in vehicle_stats.columns
            ]
            
            # Check if columns already exist before merging
            existing_vehicle_cols = [col for col in vehicle_stats.columns if col in df.columns]
            if existing_vehicle_cols:
                logger.warning(f"Found {len(existing_vehicle_cols)} existing vehicle columns. Skipping: {existing_vehicle_cols}")
                # Remove existing columns from vehicle_stats
                vehicle_stats = vehicle_stats.drop(columns=existing_vehicle_cols)
            
            # Merge back to main dataframe
            if not vehicle_stats.empty:
                df = df.merge(
                    vehicle_stats,
                    left_on='Vehicle_ID',
                    right_index=True,
                    how='left',
                    suffixes=('', '_dup')  # Add suffix to handle any remaining duplicates
                )
                
                # Remove any columns with _dup suffix
                dup_cols = [col for col in df.columns if col.endswith('_dup')]
                if dup_cols:
                    logger.warning(f"Removing {len(dup_cols)} duplicate columns created during merge")
                    df = df.drop(columns=dup_cols)
        
        return df
    
    def _create_unusual_behavior_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create indicators for unusual behavior."""
        # Unusual speed (deviation > 2 std)
        if all(col in df.columns for col in ['speed_raw', 'Driver_Velocidad_mean', 'Driver_Velocidad_std']):
            df['Is_Unusual_Speed'] = (
                np.abs(df['speed_raw'] - df['Driver_Velocidad_mean']) > 
                2 * df['Driver_Velocidad_std']
            ).astype(int)
        
        # Unusual stationary duration
        if all(col in df.columns for col in ['Stationary_Duration', 'Driver_Stationary_Duration_mean']):
            df['Is_Unusual_Stationary'] = (
                df['Stationary_Duration'] > 
                df['Driver_Stationary_Duration_mean'] * 2
            ).astype(int)
        
        # Multiple unusual indicators
        unusual_columns = [col for col in df.columns if col.startswith('Is_Unusual_')]
        if unusual_columns:
            df['Unusual_Behavior_Count'] = df[unusual_columns].sum(axis=1)
            df['Has_Multiple_Unusual'] = (df['Unusual_Behavior_Count'] >= 2).astype(int)
        
        # Combined risk score
        risk_factors = []
        if 'Is_NightTime' in df.columns:
            risk_factors.append(df['Is_NightTime'])
        if 'Is_Rare_Location' in df.columns:
            risk_factors.append(df['Is_Rare_Location'])
        if 'Has_Multiple_Unusual' in df.columns:
            risk_factors.append(df['Has_Multiple_Unusual'])
        
        if risk_factors:
            df['Behavioral_Risk_Score'] = sum(risk_factors) / len(risk_factors)
        
        return df
    
    def get_driver_profile(self, driver_name: str) -> Dict:
        """
        Get behavioral profile for a specific driver.
        
        Args:
            driver_name: Name of the driver
            
        Returns:
            Dictionary with driver statistics
        """
        if self.driver_stats is None or driver_name not in self.driver_stats.index:
            return {}
        
        profile = self.driver_stats.loc[driver_name].to_dict()
        
        # Add interpretations
        profile['risk_level'] = 'Low'
        if profile.get('Driver_Is_NightTime_mean', 0) > 0.3:
            profile['risk_level'] = 'Medium'
        if profile.get('Driver_Fuel_Diff_min', 0) < -10:
            profile['risk_level'] = 'High'
        
        return profile