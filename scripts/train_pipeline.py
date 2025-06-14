#!/usr/bin/env python
"""
Main training pipeline for fuel theft detection with optimized results storage.
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config.config import Config
from src.data.loader import DataLoader
from src.data.cleaner import DataCleaner
from src.features.builder import FeatureBuilder
from src.models.unsupervised import UnsupervisedModel
from src.models.supervised import SupervisedModel
from src.models.ensemble import EnsembleModel
from src.visualization.reports import ReportGenerator
from src.utils.logger import setup_logging, get_logger
from src.utils.helpers import save_dataframe, save_json, create_experiment_id

logger = get_logger(__name__)


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


class FuelTheftPipeline:
    """Main pipeline for fuel theft detection."""
    
    def __init__(self, config: Config):
        """
        Initialize pipeline.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.experiment_id = create_experiment_id({
            'test_size': config.model.test_size,
            'random_state': config.model.random_state,
            'resampling_method': config.model.resampling_method
        })
        
        # Create experiment directory
        self.experiment_dir = config.output_dir / f"experiment_{self.experiment_id}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting experiment: {self.experiment_id}")
    
    def run(self, data_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Run the complete pipeline.
        
        Args:
            data_path: Path to data directory. If None, uses config default
            
        Returns:
            Dictionary with pipeline results (summarized for JSON storage)
        """
        results = {}
        full_results = {}  # Store full results for processing but not for JSON
        
        # 1. Data Loading
        logger.info("=" * 60)
        logger.info("PHASE 1: DATA LOADING")
        logger.info("=" * 60)
        
        loader = DataLoader(self.config)
        data_path = data_path or self.config.data.raw_data_path
        
        raw_data = loader.load_from_directory(data_path)
        results['data_info'] = loader.get_data_info()
        
        # Save raw combined data
        save_dataframe(
            raw_data, 
            self.experiment_dir / 'data' / 'raw_combined.parquet'
        )
        
        # 2. Data Cleaning
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2: DATA CLEANING")
        logger.info("=" * 60)
        
        cleaner = DataCleaner(self.config)
        cleaned_data = cleaner.clean_data(raw_data)
        cleaned_data = cleaner.interpolate_missing_values(cleaned_data)
        
        # Validate cleaned data
        validation_issues = cleaner.validate_cleaned_data(cleaned_data)
        if validation_issues:
            logger.warning(f"Validation issues: {validation_issues}")
        
        results['cleaning_report'] = cleaner.get_cleaning_report()
        
        # Save cleaned data
        save_dataframe(
            cleaned_data,
            self.experiment_dir / 'data' / 'cleaned.parquet'
        )
        
        # 3. Feature Engineering
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 3: FEATURE ENGINEERING")
        logger.info("=" * 60)
        
        feature_builder = FeatureBuilder(self.config)
        featured_data = feature_builder.build_features(cleaned_data)
        
        # Get feature statistics
        feature_stats = feature_builder.get_feature_statistics(featured_data)
        
        # Select features for modeling
        model_features, feature_names = feature_builder.select_features_for_modeling(
            featured_data
        )
        
        results['feature_info'] = {
            'total_features': len(feature_names),
            'feature_names': feature_names,
            'feature_groups': {
                k: len(v) for k, v in feature_builder.feature_metadata.items()
            } if hasattr(feature_builder, 'feature_metadata') else {}
        }
        
        # Save featured data and statistics
        save_dataframe(
            featured_data,
            self.experiment_dir / 'data' / 'featured.parquet'
        )
        save_dataframe(
            feature_stats,
            self.experiment_dir / 'features' / 'feature_statistics.csv'
        )
        
        # 4. Unsupervised Learning
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 4: UNSUPERVISED LEARNING")
        logger.info("=" * 60)
        
        unsupervised = UnsupervisedModel(self.config)
        unsupervised_results = unsupervised.fit_predict(model_features, featured_data)
        full_results['unsupervised'] = unsupervised_results
        
        # Create summary for JSON
        results['unsupervised'] = self._summarize_unsupervised_results(unsupervised_results)
        
        # 5. Supervised Learning (if labels available)
        if 'Stationary_drain' in featured_data.columns:
            logger.info("\n" + "=" * 60)
            logger.info("PHASE 5: SUPERVISED LEARNING")
            logger.info("=" * 60)
            
            supervised = SupervisedModel(self.config)
            supervised_results = supervised.train_evaluate(
                featured_data,
                feature_names,
                target='Stationary_drain'
            )
            full_results['supervised'] = supervised_results
            
            # Create summary for JSON
            results['supervised'] = self._summarize_supervised_results(supervised_results)
            
            # Save trained models
            models_dir = self.experiment_dir / 'models'
            supervised.save_all_models(models_dir)
            supervised.save_best_model(models_dir / 'best_model.pkl')
            
            # 6. Ensemble Model
            logger.info("\n" + "=" * 60)
            logger.info("PHASE 6: ENSEMBLE MODEL")
            logger.info("=" * 60)
            
            ensemble = EnsembleModel(self.config)
            ensemble_results = ensemble.combine_predictions(
                featured_data,
                full_results['unsupervised'],
                full_results['supervised']
            )
            
            # Create summary for JSON
            results['ensemble'] = self._summarize_ensemble_results(
                ensemble_results, 
                featured_data
            )
            
            # Save final predictions
            final_predictions = featured_data[['timestamp', 'Vehicle_ID', 'driver_name', 'location', 
                                             'total_fuel', 'Fuel_Diff', 'Final_Score', 
                                             'Final_Prediction']].copy()
            save_dataframe(
                final_predictions,
                self.experiment_dir / 'predictions' / 'final_predictions.csv'
            )
        
        # 7. Generate Reports
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 7: GENERATING REPORTS")
        logger.info("=" * 60)
        
        report_gen = ReportGenerator(self.config)
        report_gen.generate_full_report(
            full_results,  # Use full results for report generation
            output_dir=self.experiment_dir / 'reports'
        )
        
        # Add metadata to results
        results['experiment_metadata'] = {
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'test_size': self.config.model.test_size,
                'random_state': self.config.model.random_state,
                'resampling_method': self.config.model.resampling_method
            },
            'output_directory': str(self.experiment_dir)
        }
        
        # Save summarized experiment results
        save_json(
            results,
            self.experiment_dir / 'experiment_results.json'
        )
        
        # Save detailed results separately if needed
        save_json(
            {
                'unsupervised_metrics': full_results.get('unsupervised', {}).get('metrics', {}),
                'supervised_metrics': full_results.get('supervised', {}).get('metrics', {}),
                'ensemble_metrics': full_results.get('ensemble', {}).get('metrics', {})
            },
            self.experiment_dir / 'detailed_metrics.json'
        )
        
        logger.info(f"\nExperiment complete! Results saved to: {self.experiment_dir}")
        
        return results
    
    def _summarize_unsupervised_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of unsupervised results for JSON storage."""
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
        
        return summary
    
    def _summarize_supervised_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of supervised results for JSON storage."""
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
        
        return summary
    
    def _summarize_ensemble_results(self, results: Dict[str, Any], 
                                   featured_data: pd.DataFrame) -> Dict[str, Any]:
        """Create summary of ensemble results for JSON storage."""
        summary = {
            'total_records': len(featured_data),
            'execution_time': results.get('execution_time', 0)
        }
        
        # Calculate detection statistics
        if 'Final_Prediction' in featured_data.columns:
            summary['total_predictions'] = int((featured_data['Final_Prediction'] == 1).sum())
            summary['detection_rate'] = float(summary['total_predictions'] / len(featured_data))
        else:
            summary['total_predictions'] = 0
            summary['detection_rate'] = 0.0
        
        # Add score statistics
        if 'Final_Score' in featured_data.columns:
            scores = featured_data['Final_Score']
            summary['score_statistics'] = {
                'mean': float(scores.mean()),
                'std': float(scores.std()),
                'min': float(scores.min()),
                'max': float(scores.max()),
                'median': float(scores.median())
            }
        
        # Add threshold information
        if 'threshold' in results:
            summary['threshold'] = float(results['threshold'])
        
        return summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train fuel theft detection models")
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Path to data directory"
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        help="Path to configuration file"
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
    
    # Load configuration
    if args.config_file:
        # TODO: Implement loading from file
        config = Config()
    else:
        config = Config.from_env()
    
    # Run pipeline
    pipeline = FuelTheftPipeline(config)
    results = pipeline.run(data_path='data/raw/')
    # results = pipeline.run(data_path=args.data_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Experiment ID: {pipeline.experiment_id}")
    print(f"Total records processed: {results['data_info']['total_records']:,}")
    print(f"Features created: {results['feature_info']['total_features']}")
    
    if 'ensemble' in results:
        print(f"Theft events detected: {results['ensemble']['total_predictions']:,}")
        print(f"Detection rate: {results['ensemble']['detection_rate']:.2%}")


if __name__ == "__main__":
    main()