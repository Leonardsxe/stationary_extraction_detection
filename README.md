# Fuel Theft Detection System

A comprehensive machine learning system for detecting fuel theft in vehicle fleets using both supervised and unsupervised learning approaches.

## Overview

This project implements a complete data science pipeline following the CRISP-DM methodology to identify potential fuel theft events from vehicle telemetry data. The system combines multiple detection approaches:

- **Unsupervised Learning**: Clustering and anomaly detection for pattern discovery
- **Supervised Learning**: Classification models for known theft patterns
- **Ensemble Methods**: Combining multiple models for robust predictions

## Features

- **Comprehensive Data Processing**: Automated data loading, cleaning, and validation
- **Advanced Feature Engineering**:
  - Temporal features (time patterns, cyclical encoding)
  - Behavioral features (driver profiles, location patterns)
  - Statistical features (rolling windows, z-scores, change detection)
- **Multiple Detection Methods**:
  - K-means clustering
  - DBSCAN for outlier detection
  - Isolation Forest for anomaly detection
  - Random Forest, XGBoost, and neural network classifiers
- **Automated Reporting**: Generate comprehensive analysis reports

## Project Structure

```
fuel_theft_detection/
├── data/
│   ├── raw/              # Original Excel files
│   ├── processed/        # Cleaned data
│   └── features/         # Feature-engineered data
├── src/
│   ├── config/          # Configuration management
│   ├── data/            # Data loading and cleaning
│   ├── features/        # Feature engineering modules
│   ├── models/          # ML models (supervised/unsupervised)
│   ├── visualization/   # Plotting and reporting
│   └── utils/           # Utility functions
├── scripts/             # Executable scripts
├── notebooks/           # Jupyter notebooks for exploration
├── tests/              # Unit tests
└── outputs/            # Model outputs and reports
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/fuel-theft-detection.git
cd fuel-theft-detection
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the complete training pipeline:

```bash
python scripts/train_pipeline.py --data-path data/raw/
```

### Step-by-Step Usage

1. **Data Preparation**:

```python
from src.data.loader import DataLoader
from src.data.cleaner import DataCleaner

# Load data
loader = DataLoader()
data = loader.load_from_directory("data/raw/")

# Clean data
cleaner = DataCleaner()
clean_data = cleaner.clean_data(data)
```

2. **Feature Engineering**:

```python
from src.features.builder import FeatureBuilder

# Build features
builder = FeatureBuilder()
featured_data = builder.build_features(clean_data)
```

3. **Model Training**:

```python
from src.models.supervised import SupervisedModel

# Train supervised models
model = SupervisedModel()
results = model.train_evaluate(featured_data, feature_names)
```

### Configuration

Modify `src/config/config.py` to adjust:

- Feature engineering parameters
- Model hyperparameters
- Anomaly detection thresholds
- Output paths

## Data Format

Input data should be Excel files with the following columns:

- `Tiempo`: Timestamp
- `Tanque Total`: Fuel level (gallons)
- `Velocidad`: Vehicle speed
- `Driver`: Driver identifier (Driver name)
- `Ubicación`: Location
- `Coordenadas`: GPS coordinates
- `Stationary_drain`: Theft label (if available, manual labeling performed by Sergio Martinez and Leonardo Lozada)

## Key Components

### Feature Engineering

The system creates three types of features:

1. **Temporal Features**:

   - Time-based patterns (hour, day, weekend)
   - Cyclical encoding for periodicity
   - Time since refuel
   - Stationary duration

2. **Behavioral Features**:
   - Driver-specific baselines
   - Location risk scores
