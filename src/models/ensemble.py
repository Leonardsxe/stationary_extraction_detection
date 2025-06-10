"""
Ensemble model combining supervised and unsupervised approaches.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
import logging

from ..config.config import Config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class EnsembleModel:
    """Combines predictions from multiple models."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize EnsembleModel.
        
        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or Config()
        
    def combine_predictions(self,
                          data: pd.DataFrame,
                          unsupervised_results: Dict[str, Any],
                          supervised_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Combine predictions from supervised and unsupervised models.
        
        Args:
            data: Full dataframe with all predictions
            unsupervised_results: Results from unsupervised models
            supervised_results: Results from supervised models (optional)
            
        Returns:
            Dictionary with ensemble results
        """
        logger.info("Creating ensemble predictions...")
        
        # Initialize final score
        data['Final_Score'] = 0
        weights_used = {}
        
        # Add unsupervised scores
        if 'Ensemble_Score' in data.columns:
            weight = 0.5 if supervised_results else 1.0
            data['Final_Score'] += data['Ensemble_Score'] * weight
            weights_used['unsupervised_ensemble'] = weight
        
        # Add supervised predictions if available
        if supervised_results and 'Theft_Probability' in data.columns:
            weight = 0.5
            data['Final_Score'] += data['Theft_Probability'] * weight
            weights_used['supervised'] = weight
        
        # Normalize final score
        if data['Final_Score'].max() > 0:
            data['Final_Score'] = data['Final_Score'] / data['Final_Score'].sum() * len(data)
            data['Final_Score'] = data['Final_Score'] / data['Final_Score'].max()
        
        # Create different threshold predictions
        thresholds = {
            'conservative': 0.8,  # High confidence
            'balanced': 0.6,      # Balanced approach
            'aggressive': 0.4     # Catch more potential thefts
        }
        
        predictions = {}
        for name, threshold in thresholds.items():
            col_name = f'Prediction_{name}'
            data[col_name] = (data['Final_Score'] > threshold).astype(int)
            predictions[name] = {
                'threshold': threshold,
                'n_predictions': data[col_name].sum(),
                'prediction_rate': data[col_name].sum() / len(data)
            }
        
        # Use balanced as default
        data['Final_Prediction'] = data['Prediction_balanced']
        
        # Analyze high-risk events
        high_risk = data[data['Final_Score'] > 0.8].copy()
        
        # Group by driver if available
        driver_risk = {}
        if 'Driver' in data.columns and len(high_risk) > 0:
            driver_risk = high_risk.groupby('Driver').agg({
                'Final_Score': ['count', 'mean'],
                'Final_Prediction': 'sum'
            }).to_dict()
        
        # Location risk
        location_risk = {}
        if 'Ubicación' in data.columns and len(high_risk) > 0:
            location_risk = high_risk.groupby('Ubicación').agg({
                'Final_Score': ['count', 'mean'],
                'Final_Prediction': 'sum'
            }).to_dict()
        
        # Time patterns
        time_risk = {}
        if 'Hour' in data.columns and len(high_risk) > 0:
            time_risk = high_risk.groupby('Hour')['Final_Score'].agg(['count', 'mean']).to_dict()
        
        results = {
            'weights_used': weights_used,
            'predictions': predictions,
            'total_predictions': data['Final_Prediction'].sum(),
            'detection_rate': data['Final_Prediction'].sum() / len(data),
            'high_risk_count': len(high_risk),
            'driver_risk': driver_risk,
            'location_risk': location_risk,
            'time_risk': time_risk,
            'score_statistics': {
                'mean': data['Final_Score'].mean(),
                'std': data['Final_Score'].std(),
                'min': data['Final_Score'].min(),
                'max': data['Final_Score'].max(),
                'quantiles': data['Final_Score'].quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
            }
        }
        
        logger.info(f"Ensemble complete: {results['total_predictions']} theft events detected "
                   f"({results['detection_rate']:.2%} detection rate)")
        
        return results
    
    def create_alert_priority(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create prioritized alerts for security team.
        
        Args:
            data: Dataframe with predictions
            
        Returns:
            Prioritized alerts dataframe
        """
        # Filter to predicted events
        alerts = data[data['Final_Prediction'] == 1].copy()
        
        if len(alerts) == 0:
            logger.warning("No alerts to prioritize")
            return pd.DataFrame()
        
        # Calculate priority score
        alerts['Priority_Score'] = 0
        
        # Factor 1: Confidence (Final_Score)
        if 'Final_Score' in alerts.columns:
            alerts['Priority_Score'] += alerts['Final_Score'] * 0.3
        
        # Factor 2: Fuel loss amount
        if 'Fuel_Diff' in alerts.columns:
            fuel_loss_normalized = alerts['Fuel_Diff'].abs() / alerts['Fuel_Diff'].abs().max()
            alerts['Priority_Score'] += fuel_loss_normalized * 0.3
        
        # Factor 3: Night time
        if 'Is_NightTime' in alerts.columns:
            alerts['Priority_Score'] += alerts['Is_NightTime'] * 0.2
        
        # Factor 4: Stationary
        if 'Is_Stationary' in alerts.columns:
            alerts['Priority_Score'] += alerts['Is_Stationary'] * 0.2
        
        # Sort by priority
        alerts = alerts.sort_values('Priority_Score', ascending=False)
        
        # Add priority categories
        alerts['Priority'] = pd.qcut(
            alerts['Priority_Score'],
            q=[0, 0.33, 0.67, 1.0],
            labels=['Low', 'Medium', 'High']
        )
        
        # Select relevant columns for alert
        alert_columns = [
            'Tiempo', 'Vehicle_ID', 'Driver', 'Ubicación',
            'Tanque Total', 'Fuel_Diff', 'Final_Score',
            'Priority_Score', 'Priority'
        ]
        
        available_columns = [col for col in alert_columns if col in alerts.columns]
        
        return alerts[available_columns]