"""
Configuration management for the fuel theft detection project.
"""
from pathlib import Path
from typing import Dict, List, Any
import os
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Data-related configuration."""
    raw_data_path: Path = Path("data/raw")
    processed_data_path: Path = Path("data/processed")
    features_data_path: Path = Path("data/features")
    
    # Column mappings
    column_mappings: Dict[str, str] = field(default_factory=lambda: {
        'Tanque total': 'total_fuel',
        'Tanque Total': 'total_fuel',
        'Calidad de señal': 'quality_signal',
        'Tanque Izquierdo': 'left_tank',
        'Batería interna': 'internal_battery',
        'Batería vehículo': 'vehicle_battery',
        'Tanque Derecho': 'right_tank',
        'Ubicación': 'location',
        'Coordenadas': 'coordinates',
        'Tiempo': 'timestamp',
        'Velocidad': 'speed_raw',
        'Altitud': 'altitude',
        'Driver': 'driver_name', 
        'Ignición': 'ignition_raw'
    })
    
    # Data types
    numeric_columns: List[str] = field(default_factory=lambda: [
        'Network Jamming', 'GNSS Jamming', 'total_fuel', 'speed_raw'
    ])
    
    # Missing value indicators
    missing_indicators: List[str] = field(default_factory=lambda: ['-----'])


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    # Temporal features
    night_hours: tuple = (22, 5)  # 10 PM to 5 AM
    
    # Threshold values
    refuel_threshold: float = 15.0  # Gallons
    stationary_loss_threshold: float = 2.5  # Gallons
    rapid_loss_rate: float = -0.5  # Gal/min
    
    # Rolling window sizes
    rolling_windows: List[int] = field(default_factory=lambda: [5, 10, 30])
    
    # Anomaly weights
    anomaly_weights: Dict[str, float] = field(default_factory=lambda: {
        'stationary_loss': 3.0,
        'night_loss': 2.0,
        'rapid_loss': 3.0,
        'high_deviation': 2.0,
        'sudden_drop': 3.0,
        'unusual_location': 1.0
    })


@dataclass
class ModelConfig:
    """Model training configuration."""
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    
    # Class imbalance handling
    resampling_method: str = 'smote'
    
    # Model hyperparameters
    logistic_params: Dict[str, Any] = field(default_factory=lambda: {
        'class_weight': 'balanced',
        'random_state': 42,
        'max_iter': 2000, 
        'solver': 'lbfgs'
    })
    
    # Random Forest parameters
    rf_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 100,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    })
    
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'random_state': 42,
        'eval_metric': 'logloss'
    })
    
    # Clustering parameters
    kmeans_max_k: int = 10
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 5
    isolation_forest_contamination: float = 0.01


@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # Logging
    log_level: str = "INFO"
    log_file: Path = Path("logs/fuel_theft.log")
    
    # Output paths
    output_dir: Path = Path("outputs")
    model_dir: Path = Path("outputs/models")
    predictions_dir: Path = Path("outputs/predictions")
    reports_dir: Path = Path("outputs/reports")
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for path in [self.output_dir, self.model_dir, 
                     self.predictions_dir, self.reports_dir]:
            path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create config from environment variables."""
        config = cls()
        
        # Override with environment variables if they exist
        if os.getenv('TEST_SIZE'):
            config.model.test_size = float(os.getenv('TEST_SIZE'))
        if os.getenv('RANDOM_STATE'):
            config.model.random_state = int(os.getenv('RANDOM_STATE'))
        
        return config


# Global config instance
config = Config.from_env()