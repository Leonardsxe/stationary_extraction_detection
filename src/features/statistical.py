"""
Statistical feature engineering for fuel theft detection.
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional, List
import logging

from ..config.config import Config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class StatisticalFeatureEngineer:
    """Creates statistical-based features."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize StatisticalFeatureEngineer.
        
        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or Config()
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all statistical features.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with statistical features
        """
        # Rolling window features
        df = self._create_rolling_features(df)
        
        # Lag features
        df = self._create_lag_features(df)
        
        # Statistical measures
        df = self._create_statistical_measures(df)
        
        # Z-scores for anomaly detection
        df = self._create_zscore_features(df)
        
        # Change point detection
        df = self._create_change_point_features(df)
        
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window statistical features."""
        if not all(col in df.columns for col in ['Vehicle_ID', 'Tanque Total']):
            logger.warning("Required columns for rolling features not found")
            return df
        
        # Sort by vehicle and time
        if 'Tiempo' in df.columns:
            df = df.sort_values(['Vehicle_ID', 'Tiempo'])
        
        rolling_columns = ['Tanque Total', 'Fuel_Diff', 'Fuel_Rate', 'Velocidad']
        rolling_columns = [col for col in rolling_columns if col in df.columns]
        
        for window in self.config.features.rolling_windows:
            for col in rolling_columns:
                # Rolling mean
                df[f'{col}_Rolling_Mean_{window}'] = (
                    df.groupby('Vehicle_ID')[col]
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(0, drop=True)
                )
                
                # Rolling std
                df[f'{col}_Rolling_Std_{window}'] = (
                    df.groupby('Vehicle_ID')[col]
                    .rolling(window=window, min_periods=2)
                    .std()
                    .reset_index(0, drop=True)
                )
                
                # Rolling min/max for certain columns
                if col in ['Tanque Total', 'Fuel_Diff']:
                    df[f'{col}_Rolling_Min_{window}'] = (
                        df.groupby('Vehicle_ID')[col]
                        .rolling(window=window, min_periods=1)
                        .min()
                        .reset_index(0, drop=True)
                    )
                    
                    df[f'{col}_Rolling_Max_{window}'] = (
                        df.groupby('Vehicle_ID')[col]
                        .rolling(window=window, min_periods=1)
                        .max()
                        .reset_index(0, drop=True)
                    )
            
            # Exponential weighted moving average
            if 'Tanque Total' in df.columns:
                df[f'Fuel_EWMA_{window}'] = (
                    df.groupby('Vehicle_ID')['Tanque Total']
                    .ewm(span=window, adjust=False)
                    .mean()
                    .reset_index(0, drop=True)
                )
        
        return df
    
    def _create_lag_features(self, df: pd.DataFrame, lags: List[int] = [1, 2, 3]) -> pd.DataFrame:
        """Create lagged features."""
        lag_columns = ['Tanque Total', 'Fuel_Diff', 'Velocidad']
        lag_columns = [col for col in lag_columns if col in df.columns]
        
        if 'Vehicle_ID' in df.columns:
            for lag in lags:
                for col in lag_columns:
                    df[f'{col}_Lag_{lag}'] = df.groupby('Vehicle_ID')[col].shift(lag)
        else:
            for lag in lags:
                for col in lag_columns:
                    df[f'{col}_Lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def _create_statistical_measures(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical measures for anomaly detection."""
        # Fuel level range in recent window
        if 'Tanque Total' in df.columns:
            for window in [10, 30]:
                if f'Tanque Total_Rolling_Min_{window}' in df.columns:
                    df[f'Fuel_Range_{window}'] = (
                        df[f'Tanque Total_Rolling_Max_{window}'] - 
                        df[f'Tanque Total_Rolling_Min_{window}']
                    )
        
        # Coefficient of variation
        for col in ['Tanque Total', 'Fuel_Rate']:
            if col in df.columns:
                for window in self.config.features.rolling_windows:
                    mean_col = f'{col}_Rolling_Mean_{window}'
                    std_col = f'{col}_Rolling_Std_{window}'
                    if mean_col in df.columns and std_col in df.columns:
                        df[f'{col}_CV_{window}'] = (
                            df[std_col] / (df[mean_col].abs() + 1e-6)
                        )
        
        # Moving average convergence divergence (MACD) for fuel level
        if 'Tanque Total' in df.columns:
            if 'Vehicle_ID' in df.columns:
                ema_12 = df.groupby('Vehicle_ID')['Tanque Total'].ewm(span=12, adjust=False).mean()
                ema_26 = df.groupby('Vehicle_ID')['Tanque Total'].ewm(span=26, adjust=False).mean()
                df['Fuel_MACD'] = ema_12.reset_index(0, drop=True) - ema_26.reset_index(0, drop=True)
            else:
                df['Fuel_MACD'] = (
                    df['Tanque Total'].ewm(span=12, adjust=False).mean() -
                    df['Tanque Total'].ewm(span=26, adjust=False).mean()
                )
        
        return df
    
    def _create_zscore_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create z-score features for anomaly detection."""
        zscore_columns = ['Tanque Total', 'Fuel_Diff', 'Fuel_Rate', 'Stationary_Duration']
        zscore_columns = [col for col in zscore_columns if col in df.columns]
        
        for col in zscore_columns:
            # Global z-score
            if df[col].std() > 0:
                df[f'{col}_ZScore'] = np.abs(stats.zscore(df[col].fillna(df[col].mean())))
            else:
                df[f'{col}_ZScore'] = 0
            
            # Local z-score (within vehicle)
            if 'Vehicle_ID' in df.columns:
                df[f'{col}_ZScore_Local'] = df.groupby('Vehicle_ID')[col].transform(
                    lambda x: np.abs(stats.zscore(x.fillna(x.mean()))) if x.std() > 0 else 0
                )
        
        # High z-score indicator
        for col in zscore_columns:
            if f'{col}_ZScore' in df.columns:
                df[f'{col}_HighZScore'] = (df[f'{col}_ZScore'] > 3).astype(int)
        
        return df
    
    def _create_change_point_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect significant changes in fuel patterns."""
        if 'Tanque Total' not in df.columns:
            return df
        
        # Fuel level change magnitude
        if 'Fuel_Diff' in df.columns:
            df['Fuel_Change_Magnitude'] = df['Fuel_Diff'].abs()
            
            # Percentile-based thresholds
            threshold_95 = df['Fuel_Change_Magnitude'].quantile(0.95)
            threshold_99 = df['Fuel_Change_Magnitude'].quantile(0.99)
            
            df['Is_Large_Change'] = (df['Fuel_Change_Magnitude'] > threshold_95).astype(int)
            df['Is_Extreme_Change'] = (df['Fuel_Change_Magnitude'] > threshold_99).astype(int)
        
        # Consecutive losses
        if 'Vehicle_ID' in df.columns and 'Fuel_Diff' in df.columns:
            df['Is_Fuel_Loss'] = (df['Fuel_Diff'] < 0).astype(int)
            
            # Count consecutive losses
            df['Consecutive_Losses'] = 0
            for vehicle in df['Vehicle_ID'].unique():
                mask = df['Vehicle_ID'] == vehicle
                vehicle_data = df.loc[mask, 'Is_Fuel_Loss']
                
                # Group consecutive losses
                loss_groups = (vehicle_data != vehicle_data.shift()).cumsum()
                consecutive_counts = vehicle_data.groupby(loss_groups).cumsum()
                
                df.loc[mask, 'Consecutive_Losses'] = consecutive_counts
        
        # Rate of change
        if all(col in df.columns for col in ['Fuel_Diff', 'Time_Diff']):
            # Second derivative (acceleration of fuel change)
            if 'Vehicle_ID' in df.columns:
                df['Fuel_Acceleration'] = df.groupby('Vehicle_ID')['Fuel_Rate'].diff()
            else:
                df['Fuel_Acceleration'] = df['Fuel_Rate'].diff()
            
            # Sharp deceleration indicator
            df['Is_Sharp_Deceleration'] = (df['Fuel_Acceleration'] < -0.1).astype(int)
        
        return df
    
    def calculate_entropy(self, df: pd.DataFrame, column: str, window: int = 20) -> pd.Series:
        """
        Calculate entropy of a column over rolling windows.
        
        Args:
            df: Input dataframe
            column: Column name to calculate entropy for
            window: Window size
            
        Returns:
            Series with entropy values
        """
        def _entropy(x):
            """Calculate Shannon entropy."""
            if len(x) == 0:
                return 0
            
            # Discretize values into bins
            bins = np.histogram_bin_edges(x.dropna(), bins='auto')
            if len(bins) <= 1:
                return 0
            
            hist, _ = np.histogram(x.dropna(), bins=bins)
            probs = hist / hist.sum()
            probs = probs[probs > 0]  # Remove zero probabilities
            
            return -np.sum(probs * np.log2(probs))
        
        if column not in df.columns:
            logger.warning(f"Column '{column}' not found for entropy calculation")
            return pd.Series(index=df.index, dtype=float)
        
        if 'Vehicle_ID' in df.columns:
            entropy_series = (
                df.groupby('Vehicle_ID')[column]
                .rolling(window=window, min_periods=window//2)
                .apply(_entropy, raw=False)
                .reset_index(0, drop=True)
            )
        else:
            entropy_series = df[column].rolling(window=window, min_periods=window//2).apply(_entropy, raw=False)
        
        return entropy_series