#!/usr/bin/env python
"""
Prediction script for fuel theft detection on new data.
"""
import argparse
import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config.config import Config
from src.data.loader import DataLoader
from src.data.cleaner import DataCleaner
from src.features.builder import FeatureBuilder
from src.utils.logger import setup_logging, get_logger
from src.utils.helpers import load_model, save_dataframe, load_json

# Suppress warnings
warnings.filterwarnings('ignore')

logger = get_logger(__name__)


class FuelTheftPredictor:
    """Handles prediction on new data using trained models."""
    
    def __init__(self, model_path: Path, config: Config = None):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to saved model file or directory
            config: Configuration object
        """
        self.config = config or Config()
        self.model_data = None
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        # Load model
        self._load_model(model_path)
        
    def _load_model(self, model_path: Path):
        """Load trained model and associated data."""
        if model_path.is_dir():
            # Load best model from directory
            model_file = model_path / 'best_model.pkl'
            if not model_file.exists():
                raise FileNotFoundError(f"Best model not found in {model_path}")
        else:
            model_file = model_path
        
        logger.info(f"Loading model from {model_file}")
        self.model_data = load_model(model_file)
        
        # Extract components
        self.model = self.model_data['model']
        self.scaler = self.model_data['scaler']
        self.feature_names = self.model_data.get('feature_names', [])
        self.model_name = self.model_data.get('model_name', 'Unknown')
        
        logger.info(f"Loaded {self.model_name} model with {len(self.feature_names)} features")
    
    def predict(self, data_path: Path, output_path: Path = None) -> pd.DataFrame:
        """
        Make predictions on new data.
        
        Args:
            data_path: Path to new data (file or directory)
            output_path: Path to save predictions (optional)
            
        Returns:
            DataFrame with predictions
        """
        logger.info("Starting prediction pipeline...")
        
        # 1. Load data
        logger.info("Loading data...")
        loader = DataLoader(self.config)
        
        if data_path.is_dir():
            data = loader.load_from_directory(data_path)
        else:
            # Load single file
            data = loader.load_excel_files([data_path])
        
        logger.info(f"Loaded {len(data)} records")
        
        # 2. Clean data
        logger.info("Cleaning data...")
        cleaner = DataCleaner(self.config)
        clean_data = cleaner.clean_data(data)
        clean_data = cleaner.interpolate_missing_values(clean_data)
        
        # 3. Engineer features
        logger.info("Engineering features...")
        feature_builder = FeatureBuilder(self.config)
        featured_data = feature_builder.build_features(clean_data)
        
        # 4. Prepare features for prediction
        logger.info("Preparing features for prediction...")
        X = self._prepare_features(featured_data)
        
        # 5. Make predictions
        logger.info("Making predictions...")
        predictions = self._make_predictions(X)
        
        # 6. Add predictions to data
        featured_data['Predicted_Theft'] = predictions['predictions']
        featured_data['Theft_Probability'] = predictions['probabilities']
        featured_data['Theft_Risk'] = predictions['risk_level']
        
        # 7. Create output dataframe
        output_columns = [
            'timestamp', 'Vehicle_ID', 'driver_name', 'location',
            'total_fuel', 'Fuel_Diff', 'speed_raw', 'Is_Stationary',
            'Anomaly_Score', 'Predicted_Theft', 'Theft_Probability', 'Theft_Risk'
        ]
        
        # Filter to available columns
        available_columns = [col for col in output_columns if col in featured_data.columns]
        result_df = featured_data[available_columns].copy()
        
        # Filter to high-risk events
        high_risk = result_df[result_df['Predicted_Theft'] == 1].copy()
        
        logger.info(f"\nPrediction Summary:")
        logger.info(f"Total records: {len(result_df):,}")
        logger.info(f"Predicted theft events: {len(high_risk):,} ({len(high_risk)/len(result_df)*100:.2f}%)")
        
        # Risk level distribution
        if 'Theft_Risk' in result_df.columns:
            risk_dist = result_df['Theft_Risk'].value_counts()
            logger.info("\nRisk Distribution:")
            for risk, count in risk_dist.items():
                logger.info(f"  {risk}: {count:,} ({count/len(result_df)*100:.1f}%)")
        
        # 8. Save results if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save all predictions
            all_predictions_path = output_path.parent / f"all_{output_path.name}"
            save_dataframe(result_df, all_predictions_path)
            logger.info(f"\nSaved all predictions to: {all_predictions_path}")
            
            # Save high-risk events only
            if len(high_risk) > 0:
                save_dataframe(high_risk, output_path)
                logger.info(f"Saved high-risk events to: {output_path}")
            else:
                logger.info("No high-risk events to save")
        
        return result_df
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features to match training data."""
        # Get available features
        available_features = []
        missing_features = []
        
        for feature in self.feature_names:
            if feature in data.columns:
                available_features.append(feature)
            else:
                missing_features.append(feature)
        
        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features from training: {missing_features[:5]}...")
            logger.warning("These will be filled with zeros")
        
        # Create feature matrix
        X = pd.DataFrame(index=data.index)
        
        # Add available features
        for feature in self.feature_names:
            if feature in data.columns:
                X[feature] = data[feature]
            else:
                # Fill missing features with zero
                X[feature] = 0
        
        # Handle missing values
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                X[col].fillna(X[col].median() if X[col].dtype in ['float64', 'int64'] else 0, inplace=True)
        
        logger.info(f"Prepared {len(X.columns)} features for prediction")
        
        return X
    
    def _make_predictions(self, X: pd.DataFrame) -> dict:
        """Make predictions using the loaded model."""
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Get probabilities if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_scaled)[:, 1]
        else:
            probabilities = predictions.astype(float)
        
        # Assign risk levels
        risk_levels = []
        for prob in probabilities:
            if prob >= 0.8:
                risk_levels.append('High')
            elif prob >= 0.5:
                risk_levels.append('Medium')
            elif prob >= 0.3:
                risk_levels.append('Low')
            else:
                risk_levels.append('Very Low')
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'risk_level': risk_levels
        }
    
    def evaluate_on_test_data(self, data_path: Path, true_labels_column: str = 'Stationary_drain') -> dict:
        """
        Evaluate model performance on test data with known labels.
        
        Args:
            data_path: Path to test data
            true_labels_column: Column name with true labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Make predictions
        predictions_df = self.predict(data_path)
        
        # Check if true labels exist
        if true_labels_column not in predictions_df.columns:
            logger.warning(f"True labels column '{true_labels_column}' not found in data")
            return {}
        
        # Get true labels and predictions
        y_true = predictions_df[true_labels_column]
        y_pred = predictions_df['Predicted_Theft']
        y_proba = predictions_df['Theft_Probability']
        
        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix, classification_report
        )
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0,
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }
        
        # Print results
        logger.info("\n" + "="*60)
        logger.info("EVALUATION RESULTS ON TEST DATA")
        logger.info("="*60)
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Test samples: {len(y_true):,}")
        logger.info(f"Positive samples: {y_true.sum():,} ({y_true.sum()/len(y_true)*100:.2f}%)")
        logger.info("\nPerformance Metrics:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
        logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        logger.info("\nConfusion Matrix:")
        logger.info("  Predicted")
        logger.info("    0    1")
        cm = metrics['confusion_matrix']
        logger.info(f"0 {cm[0][0]:5d} {cm[0][1]:5d}")
        logger.info(f"1 {cm[1][0]:5d} {cm[1][1]:5d}")
        
        return metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Predict fuel theft on new data")
    parser.add_argument(
        "data_path",
        type=Path,
        help="Path to new data file or directory"
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to saved model file or directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save predictions"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate model performance (requires true labels)"
    )
    parser.add_argument(
        "--label-column",
        default="Stationary_drain",
        help="Column name with true labels for evaluation"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    
    # Initialize predictor
    predictor = FuelTheftPredictor(args.model)
    
    # Make predictions or evaluate
    if args.evaluate:
        metrics = predictor.evaluate_on_test_data(args.data_path, args.label_column)
        
        # Save evaluation results
        if args.output:
            eval_path = args.output.parent / f"evaluation_{args.output.stem}.json"
            from src.utils.helpers import save_json
            save_json(metrics, eval_path)
            logger.info(f"\nSaved evaluation results to: {eval_path}")
    else:
        predictor.predict(args.data_path, args.output)


if __name__ == "__main__":
    main()