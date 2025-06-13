#!/usr/bin/env python
"""
Compare performance of different trained models on test data.
"""
import argparse
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.predict import FuelTheftPredictor
from src.utils.logger import setup_logging, get_logger
from src.utils.helpers import save_json, save_dataframe

logger = get_logger(__name__)


def compare_models(models_dir: Path, test_data_path: Path, output_dir: Path = None):
    """
    Compare all models in a directory on the same test data.
    
    Args:
        models_dir: Directory containing saved models
        test_data_path: Path to test data
        output_dir: Directory to save comparison results
    """
    logger.info("="*60)
    logger.info("MODEL COMPARISON ON TEST DATA")
    logger.info("="*60)
    
    # Find all model files
    model_files = list(models_dir.glob("*.pkl"))
    
    if not model_files:
        logger.error(f"No model files found in {models_dir}")
        return
    
    logger.info(f"Found {len(model_files)} models to compare")
    
    # Results storage
    comparison_results = []
    
    # Test each model
    for model_file in model_files:
        if model_file.name == 'best_model.pkl':
            continue  # Skip duplicate of best model
        
        logger.info(f"\n{'='*40}")
        logger.info(f"Testing: {model_file.name}")
        logger.info("="*40)
        
        try:
            # Load model and make predictions
            predictor = FuelTheftPredictor(model_file)
            
            # Evaluate on test data
            metrics = predictor.evaluate_on_test_data(test_data_path)
            
            # Store results
            result = {
                'model': model_file.stem,
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1_score': metrics.get('f1_score', 0),
                'roc_auc': metrics.get('roc_auc', 0)
            }
            comparison_results.append(result)
            
        except Exception as e:
            logger.error(f"Error testing {model_file.name}: {str(e)}")
            continue
    
    # Create comparison dataframe
    if comparison_results:
        df = pd.DataFrame(comparison_results)
        df = df.sort_values('f1_score', ascending=False)
        
        # Print comparison table
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON SUMMARY")
        logger.info("="*60)
        print("\n" + df.to_string(index=False, float_format='%.4f'))
        
        # Save results
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save CSV
            csv_path = output_dir / 'model_comparison.csv'
            df.to_csv(csv_path, index=False)
            logger.info(f"\nSaved comparison to: {csv_path}")
            
            # Save detailed JSON
            json_path = output_dir / 'model_comparison_detailed.json'
            save_json({
                'test_data': str(test_data_path),
                'models_tested': len(comparison_results),
                'results': comparison_results,
                'best_model': df.iloc[0].to_dict() if len(df) > 0 else {}
            }, json_path)
        
        # Identify best model
        best_model = df.iloc[0]
        logger.info(f"\nBest Model: {best_model['model']}")
        logger.info(f"  F1-Score: {best_model['f1_score']:.4f}")
        logger.info(f"  Precision: {best_model['precision']:.4f}")
        logger.info(f"  Recall: {best_model['recall']:.4f}")
        
        # Warning if F1 is perfect
        if best_model['f1_score'] >= 0.99:
            logger.warning("\n⚠️  WARNING: Near-perfect F1 score may indicate:")
            logger.warning("   - Model is overfitting")
            logger.warning("   - Test data is too similar to training data")
            logger.warning("   - Consider testing on completely independent data")
    
    else:
        logger.error("No models were successfully tested")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Compare model performance on test data")
    parser.add_argument(
        "--models-dir",
        type=Path,
        required=True,
        help="Directory containing saved models"
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        required=True,
        help="Path to test data file or directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save comparison results"
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
    
    # Run comparison
    compare_models(args.models_dir, args.test_data, args.output_dir)


if __name__ == "__main__":
    main()