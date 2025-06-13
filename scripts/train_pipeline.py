#!/usr/bin/env python
"""
Main training pipeline for fuel theft detection.
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

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
            Dictionary with pipeline results
        """
        results = {}
        
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
            'feature_metadata': feature_builder.feature_metadata
        }
        
        # Save featured data
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
        
        results['unsupervised'] = unsupervised_results
        
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
            
            results['supervised'] = supervised_results
            
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
                unsupervised_results,
                supervised_results
            )
            
            results['ensemble'] = ensemble_results
            
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
            results,
            output_dir=self.experiment_dir / 'reports'
        )
        
        # Save experiment results
        save_json(
            results,
            self.experiment_dir / 'experiment_results.json'
        )
        
        logger.info(f"\nExperiment complete! Results saved to: {self.experiment_dir}")
        
        return results


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