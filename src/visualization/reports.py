"""
Report generation for fuel theft detection results.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, Any
import logging
from datetime import datetime

from ..config.config import Config
from ..utils.logger import get_logger
from ..utils.helpers import save_json

logger = get_logger(__name__)


class ReportGenerator:
    """Generates comprehensive analysis reports."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize ReportGenerator.
        
        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or Config()
        
    def generate_full_report(self, 
                           results: Dict[str, Any],
                           output_dir: Optional[Path] = None) -> None:
        """
        Generate complete analysis report.
        
        Args:
            results: Dictionary with all pipeline results
            output_dir: Directory to save report files
        """
        output_dir = Path(output_dir) if output_dir else self.config.reports_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating report in {output_dir}")
        
        # Create report sections
        self._create_summary_page(results, output_dir)
        self._create_data_quality_report(results, output_dir)
        self._create_feature_analysis(results, output_dir)
        self._create_model_performance_report(results, output_dir)
        self._create_detection_analysis(results, output_dir)
        
        # Generate markdown report
        self._create_markdown_report(results, output_dir)
        
        logger.info("Report generation complete")
    
    def _create_summary_page(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Create executive summary visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Fuel Theft Detection - Executive Summary', fontsize=16)
        
        # 1. Data overview
        ax = axes[0, 0]
        if 'data_info' in results:
            info = results['data_info']
            labels = ['Total Records', 'Vehicles', 'Drivers']
            values = [
                info.get('total_records', 0),
                info.get('vehicles', 0),
                len(set(str(d) for d in info.get('drivers', []))) if 'drivers' in info else 0
            ]
            ax.bar(labels, values)
            ax.set_title('Data Overview')
            ax.set_ylabel('Count')
            
            # Add value labels on bars
            for i, v in enumerate(values):
                ax.text(i, v + max(values)*0.01, f'{v:,}', ha='center')
        
        # 2. Detection results
        ax = axes[0, 1]
        if 'ensemble' in results:
            ensemble = results['ensemble']
            detection_data = {
                'Total Events': ensemble.get('total_predictions', 0),
                'Detection Rate': ensemble.get('detection_rate', 0) * 100,
                'High Risk': ensemble.get('high_risk_count', 0)
            }
            
            # Create bar plot
            x = range(len(detection_data))
            values = list(detection_data.values())
            ax.bar(x, values)
            ax.set_xticks(x)
            ax.set_xticklabels(detection_data.keys())
            ax.set_title('Detection Summary')
            
            # Add value labels
            for i, v in enumerate(values):
                label = f'{v:,.0f}' if i != 1 else f'{v:.2f}%'
                ax.text(i, v + max(values)*0.01, label, ha='center')
        
        # 3. Model performance
        ax = axes[1, 0]
        if 'supervised' in results and 'model_results' in results['supervised']:
            model_results = results['supervised']['model_results']
            
            # Extract F1 scores
            models = list(model_results.keys())
            f1_scores = [model_results[m].get('f1', 0) for m in models]
            
            # Create horizontal bar plot
            y_pos = np.arange(len(models))
            ax.barh(y_pos, f1_scores)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(models)
            ax.set_xlabel('F1 Score')
            ax.set_title('Model Performance Comparison')
            ax.set_xlim(0, 1)
            
            # Add value labels
            for i, v in enumerate(f1_scores):
                ax.text(v + 0.01, i, f'{v:.3f}', va='center')
        
        # 4. Feature importance
        ax = axes[1, 1]
        if 'supervised' in results and 'feature_importance' in results['supervised']:
            importance = results['supervised']['feature_importance']
            if not importance.empty and 'Mean_Importance' in importance.columns:
                top_features = importance.nlargest(10, 'Mean_Importance')
                ax.barh(range(len(top_features)), top_features['Mean_Importance'])
                ax.set_yticks(range(len(top_features)))
                ax.set_yticklabels(top_features.index)
                ax.set_xlabel('Importance')
                ax.set_title('Top 10 Most Important Features')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'executive_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_data_quality_report(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Create data quality visualizations."""
        if 'cleaning_report' not in results:
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        cleaning = results['cleaning_report']
        
        # Create pie chart of data retention
        sizes = [cleaning.get('final_rows', 0), cleaning.get('rows_removed', 0)]
        labels = ['Retained', 'Removed']
        colors = ['#2ecc71', '#e74c3c']
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Data Cleaning Results')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'data_quality.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_feature_analysis(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Create feature analysis visualizations."""
        if 'feature_info' not in results:
            return
        
        feature_info = results['feature_info']
        
        # Feature categories
        if 'feature_metadata' in feature_info:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            categories = feature_info['feature_metadata']
            cat_names = list(categories.keys())
            cat_counts = [len(features) for features in categories.values()]
            
            ax.bar(cat_names, cat_counts)
            ax.set_xlabel('Feature Category')
            ax.set_ylabel('Number of Features')
            ax.set_title('Features by Category')
            
            # Add value labels
            for i, v in enumerate(cat_counts):
                ax.text(i, v + 0.5, str(v), ha='center')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'feature_categories.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_model_performance_report(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Create model performance visualizations."""
        if 'supervised' not in results:
            return
        
        supervised = results['supervised']
        
        # Performance metrics comparison
        if 'model_results' in supervised:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            model_results = supervised['model_results']
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            
            # Prepare data for grouped bar plot
            models = list(model_results.keys())
            x = np.arange(len(models))
            width = 0.2
            
            for i, metric in enumerate(metrics):
                values = [model_results[m].get(metric, 0) for m in models]
                ax.bar(x + i*width, values, width, label=metric.capitalize())
            
            ax.set_xlabel('Model')
            ax.set_ylabel('Score')
            ax.set_title('Model Performance Metrics')
            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.legend()
            ax.set_ylim(0, 1.1)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'model_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_detection_analysis(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Create detection analysis visualizations."""
        if 'ensemble' not in results:
            return
        
        ensemble = results['ensemble']
        
        # Create multi-panel figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Theft Detection Analysis', fontsize=16)
        
        # 1. Score distribution
        ax = axes[0, 0]
        if 'score_statistics' in ensemble:
            stats = ensemble['score_statistics']
            
            # Create a mock distribution visualization
            ax.text(0.1, 0.9, f"Score Statistics:", transform=ax.transAxes, fontweight='bold')
            ax.text(0.1, 0.8, f"Mean: {stats['mean']:.3f}", transform=ax.transAxes)
            ax.text(0.1, 0.7, f"Std: {stats['std']:.3f}", transform=ax.transAxes)
            ax.text(0.1, 0.6, f"Min: {stats['min']:.3f}", transform=ax.transAxes)
            ax.text(0.1, 0.5, f"Max: {stats['max']:.3f}", transform=ax.transAxes)
            
            ax.text(0.1, 0.3, "Quantiles:", transform=ax.transAxes, fontweight='bold')
            for q, v in stats.get('quantiles', {}).items():
                ax.text(0.1, 0.2 - q*0.05, f"{q}: {v:.3f}", transform=ax.transAxes)
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title('Score Distribution Statistics')
        
        # 2. Predictions by threshold
        ax = axes[0, 1]
        if 'predictions' in ensemble:
            predictions = ensemble['predictions']
            thresholds = []
            counts = []
            
            for name, pred_info in predictions.items():
                thresholds.append(name)
                counts.append(pred_info['n_predictions'])
            
            ax.bar(thresholds, counts)
            ax.set_xlabel('Threshold Type')
            ax.set_ylabel('Number of Detections')
            ax.set_title('Detections by Threshold')
            
            # Add value labels
            for i, v in enumerate(counts):
                ax.text(i, v + max(counts)*0.01, str(v), ha='center')
        
        # 3. Hour distribution (if available)
        ax = axes[1, 0]
        if 'time_risk' in ensemble and ensemble['time_risk']:
            time_risk = ensemble['time_risk']
            if 'count' in time_risk:
                hours = list(time_risk['count'].keys())
                counts = list(time_risk['count'].values())
                
                ax.bar(hours, counts)
                ax.set_xlabel('Hour of Day')
                ax.set_ylabel('High Risk Events')
                ax.set_title('High Risk Events by Hour')
                ax.set_xticks(range(0, 24, 2))
        else:
            ax.text(0.5, 0.5, 'No time pattern data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Time Pattern Analysis')
        
        # 4. Summary statistics
        ax = axes[1, 1]
        summary_text = f"""
Detection Summary:
━━━━━━━━━━━━━━━━━━━━━
Total Predictions: {ensemble.get('total_predictions', 0):,}
Detection Rate: {ensemble.get('detection_rate', 0)*100:.2f}%
High Risk Events: {ensemble.get('high_risk_count', 0):,}

Threshold Used: Balanced
Confidence Level: Medium
        """
        
        ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, 
               fontfamily='monospace', verticalalignment='center')
        ax.axis('off')
        ax.set_title('Detection Summary')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'detection_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_markdown_report(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Create markdown report."""
        report_lines = [
            "# Fuel Theft Detection Report",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n## Executive Summary\n"
        ]
        
        # Data overview
        if 'data_info' in results:
            info = results['data_info']
            report_lines.extend([
                "### Data Overview",
                f"- Total Records: {info.get('total_records', 0):,}",
                f"- Vehicles: {info.get('vehicles', 0)}",
                f"- Date Range: {info.get('date_range', {}).get('start', 'N/A')} to {info.get('date_range', {}).get('end', 'N/A')}",
                ""
            ])
        
        # Detection results
        if 'ensemble' in results:
            ensemble = results['ensemble']
            report_lines.extend([
                "### Detection Results",
                f"- Total Theft Events Detected: {ensemble.get('total_predictions', 0):,}",
                f"- Detection Rate: {ensemble.get('detection_rate', 0)*100:.2f}%",
                f"- High Risk Events: {ensemble.get('high_risk_count', 0):,}",
                ""
            ])
        
        # Model performance
        if 'supervised' in results:
            supervised = results['supervised']
            report_lines.extend([
                "### Model Performance",
                f"- Best Model: {supervised.get('best_model', 'N/A')}",
                f"- F1 Score: {supervised.get('best_score', 0):.4f}",
                ""
            ])
        
        # Feature importance
        if 'feature_info' in results:
            report_lines.extend([
                "### Feature Engineering",
                f"- Total Features Created: {results['feature_info'].get('total_features', 0)}",
                "- Feature Categories:",
            ])
            
            if 'feature_metadata' in results['feature_info']:
                for category, features in results['feature_info']['feature_metadata'].items():
                    report_lines.append(f"  - {category}: {len(features)} features")
            
            report_lines.append("")
        
        # Recommendations
        report_lines.extend([
            "## Recommendations\n",
            "1. **Immediate Actions:**",
            "   - Review high-risk events flagged by the system",
            "   - Focus on night-time stationary events with significant fuel loss",
            "   - Investigate drivers and locations with multiple detections",
            "",
            "2. **Long-term Improvements:**",
            "   - Implement real-time monitoring for immediate alerts",
            "   - Enhance GPS tracking to reduce location data gaps",
            "   - Regular model retraining with new data",
            "",
            "3. **Operational Changes:**",
            "   - Increase security at high-risk locations",
            "   - Implement driver rotation for high-risk routes",
            "   - Regular fuel audits for suspicious patterns",
            ""
        ])
        
        # Write report
        report_path = output_dir / 'analysis_report.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Markdown report saved to {report_path}")