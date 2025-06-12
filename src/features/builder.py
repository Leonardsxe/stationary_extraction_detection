"""
Feature engineering pipeline orchestrator.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

from .temporal import TemporalFeatureEngineer
from .behavioral import BehavioralFeatureEngineer
from .statistical import StatisticalFeatureEngineer
from ..config.config import Config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class FeatureBuilder:
    """Orchestrates the feature engineering pipeline."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize FeatureBuilder.
        
        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or Config()
        self.feature_engineers = {
            'temporal': TemporalFeatureEngineer(config),
            'behavioral': BehavioralFeatureEngineer(config),
            'statistical': StatisticalFeatureEngineer(config)
        }
        self.feature_metadata = {}
        
    def build_features(self, df: pd.DataFrame, 
                      feature_sets: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Build all features or specified feature sets.
        
        Args:
            df: Input dataframe
            feature_sets: List of feature sets to build. If None, builds all.
                         Options: ['temporal', 'behavioral', 'statistical']
        
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering pipeline...")
        
        # Default to all feature sets
        if feature_sets is None:
            feature_sets = list(self.feature_engineers.keys())
        
        # Create a copy to avoid modifying original
        df_features = df.copy()
        
        # Check for duplicate columns in input
        if df_features.columns.duplicated().any():
            n_duplicates = df_features.columns.duplicated().sum()
            logger.warning(f"Input data has {n_duplicates} duplicate columns. Removing duplicates...")
            df_features = df_features.loc[:, ~df_features.columns.duplicated(keep='first')]
        
        # Track original columns
        original_columns = set(df_features.columns)
        
        # Apply each feature engineering step
        for feature_set in feature_sets:
            if feature_set in self.feature_engineers:
                logger.info(f"Building {feature_set} features...")
                
                engineer = self.feature_engineers[feature_set]
                df_features = engineer.create_features(df_features)
                
                # Check for duplicates after each step
                if df_features.columns.duplicated().any():
                    n_duplicates = df_features.columns.duplicated().sum()
                    duplicate_cols = df_features.columns[df_features.columns.duplicated()].tolist()
                    logger.warning(f"Feature engineering step '{feature_set}' created {n_duplicates} duplicate columns: {duplicate_cols}")
                    
                    # Remove duplicates, keeping the first occurrence
                    df_features = df_features.loc[:, ~df_features.columns.duplicated(keep='first')]
                
                # Track new features
                new_features = set(df_features.columns) - original_columns
                self.feature_metadata[feature_set] = list(new_features - set(sum(self.feature_metadata.values(), [])))
                
                logger.info(f"Created {len(self.feature_metadata[feature_set])} {feature_set} features")
            else:
                logger.warning(f"Unknown feature set: {feature_set}")
        
        # Create anomaly scores
        df_features = self._create_anomaly_scores(df_features)
        
        # Final duplicate check
        if df_features.columns.duplicated().any():
            n_duplicates = df_features.columns.duplicated().sum()
            logger.warning(f"Final dataframe has {n_duplicates} duplicate columns. Removing...")
            df_features = df_features.loc[:, ~df_features.columns.duplicated(keep='first')]
        
        # Log feature summary
        total_features = len(df_features.columns) - len(original_columns)
        logger.info(f"Feature engineering complete. Created {total_features} new features")
        
        return df_features
    
    def _create_anomaly_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite anomaly scores."""
        logger.debug("Creating anomaly scores...")
        
        # Define anomaly conditions
        anomaly_conditions = {
            'stationary_loss': (
                (df.get('Is_Stationary', 0) == 1) & 
                (df.get('Fuel_Diff', 0) < -self.config.features.stationary_loss_threshold)
            ),
            'night_loss': (
                (df.get('Is_NightTime', 0) == 1) & 
                (df.get('Fuel_Diff', 0) < -3)
            ),
            'rapid_loss': (
                (df.get('Fuel_Rate', 0) < self.config.features.rapid_loss_rate) & 
                (df.get('Is_Stationary', 0) == 1)
            ),
            'high_deviation': (
                df.get('Fuel_Deviation_From_Driver', 0).abs() > 3
            ),
            'sudden_drop': (
                df.get('Is_Sudden_Drop', 0) == 1
            ),
            'unusual_location': (
                df.get('Location_Risk_Score', 1) < 0.1
            )
        }
        
        # Calculate weighted anomaly score
        df['Anomaly_Score'] = 0
        
        for condition_name, condition in anomaly_conditions.items():
            weight = self.config.features.anomaly_weights.get(condition_name, 1.0)
            # Ensure condition is boolean
            condition_value = condition.fillna(False).astype(int)
            df['Anomaly_Score'] += condition_value * weight
            
            # Store individual anomaly flags
            df[f'Anomaly_{condition_name}'] = condition_value
        
        # Normalize anomaly score
        if df['Anomaly_Score'].max() > 0:
            df['Anomaly_Score'] = df['Anomaly_Score'] / df['Anomaly_Score'].max()
        
        # Add to metadata
        anomaly_features = ['Anomaly_Score'] + [f'Anomaly_{name}' for name in anomaly_conditions.keys()]
        self.feature_metadata['anomaly'] = anomaly_features
        
        logger.info(f"Created {len(anomaly_features)} anomaly features")
        
        return df
    
    def get_feature_names(self, feature_sets: Optional[List[str]] = None) -> List[str]:
        """
        Get list of feature names for specified feature sets.
        
        Args:
            feature_sets: List of feature sets. If None, returns all.
            
        Returns:
            List of feature names
        """
        if feature_sets is None:
            feature_sets = list(self.feature_metadata.keys())
        
        features = []
        for feature_set in feature_sets:
            features.extend(self.feature_metadata.get(feature_set, []))
        
        return features
    
    def select_features_for_modeling(self, df: pd.DataFrame, 
                                   exclude_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select and prepare features for modeling.
        
        Args:
            df: DataFrame with all features
            exclude_columns: Columns to exclude from features
            
        Returns:
            Tuple of (feature dataframe, feature names)
        """
        logger.info("Selecting features for modeling...")
        
        # Default columns to exclude
        default_exclude = [
            'Vehicle_ID', 'driver_name', 'timestamp', 'Source_File', 
            'coordinates', 'location', 'Latitude', 'Longitude'
        ]
        
        if exclude_columns:
            default_exclude.extend(exclude_columns)
        
        # Get numeric columns only
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove excluded columns
        feature_columns = [col for col in numeric_columns if col not in default_exclude]
        
        # Handle missing values
        features_df = df[feature_columns].copy()
        
        # Fill missing values with appropriate strategies
        for col in features_df.columns:
            na_count = features_df[col].isna().sum()
            if na_count > 0:
                # Use median for numeric features
                median_value = features_df[col].median()
                features_df[col].fillna(median_value, inplace=True)
                logger.debug(f"Filled {na_count} missing values in '{col}' with median: {median_value:.2f}")
        
        logger.info(f"Selected {len(feature_columns)} features for modeling")
        
        return features_df, feature_columns
    
    def get_feature_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate statistics for all engineered features.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with feature statistics
        """
        # Get all engineered features
        all_features = []
        for features in self.feature_metadata.values():
            all_features.extend(features)
        
        # Remove duplicates from feature list
        all_features = list(dict.fromkeys(all_features))  # Preserves order
        
        # Filter to existing columns
        existing_features = [f for f in all_features if f in df.columns]
        
        if not existing_features:
            logger.warning("No engineered features found in dataframe")
            return pd.DataFrame()
        
        # Handle duplicate columns in dataframe
        if df.columns.duplicated().any():
            logger.warning(f"Found {df.columns.duplicated().sum()} duplicate columns. Removing duplicates...")
            df = df.loc[:, ~df.columns.duplicated(keep='first')]
        
        # Ensure features are still in dataframe after duplicate removal
        existing_features = [f for f in existing_features if f in df.columns]
        
        if not existing_features: # Re-check after potential column removal
            logger.warning("No engineered features found in dataframe after duplicate column removal for statistics.")
            return pd.DataFrame()
            
        # Calculate statistics
        stats = df[existing_features].describe().T
        
        # Add additional statistics
        stats['missing_count'] = df[existing_features].isna().sum()
        stats['missing_pct'] = stats['missing_count'] / len(df) * 100
        stats['unique_count'] = df[existing_features].nunique()
        
        numeric_features_for_stats = df[existing_features].select_dtypes(include=np.number).columns
        stats['skewness'] = df[numeric_features_for_stats].skew()
        stats['kurtosis'] = df[numeric_features_for_stats].kurtosis()
        
        return stats.round(3)