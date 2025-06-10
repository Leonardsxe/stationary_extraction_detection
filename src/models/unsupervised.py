"""
Unsupervised learning models for fuel theft detection.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from ..config.config import Config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class UnsupervisedModel:
    """Handles unsupervised learning for anomaly detection."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize UnsupervisedModel.
        
        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or Config()
        self.scaler = None
        self.models = {}
        self.results = {}
        
    def fit_predict(self, X: pd.DataFrame, full_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Fit unsupervised models and make predictions.
        
        Args:
            X: Feature matrix
            full_data: Full dataframe with all columns
            
        Returns:
            Dictionary with results
        """
        logger.info("Starting unsupervised learning...")
        
        # Scale features
        X_scaled = self._scale_features(X)
        
        # PCA analysis
        pca_results = self._perform_pca(X_scaled)
        
        # Clustering
        kmeans_results = self._apply_kmeans(X_scaled, full_data)
        dbscan_results = self._apply_dbscan(X_scaled, full_data)
        
        # Anomaly detection
        iso_forest_results = self._apply_isolation_forest(X_scaled, full_data)
        
        # Combine results
        ensemble_results = self._create_ensemble_score(full_data)
        
        # Compile all results
        results = {
            'pca': pca_results,
            'kmeans': kmeans_results,
            'dbscan': dbscan_results,
            'isolation_forest': iso_forest_results,
            'ensemble': ensemble_results,
            'n_anomalies': ensemble_results['n_anomalies'],
            'anomaly_rate': ensemble_results['anomaly_rate']
        }
        
        self.results = results
        return results
    
    def _scale_features(self, X: pd.DataFrame) -> np.ndarray:
        """Scale features using robust scaling."""
        logger.debug("Scaling features...")
        
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled
    
    def _perform_pca(self, X_scaled: np.ndarray) -> Dict[str, Any]:
        """Perform PCA analysis."""
        logger.debug("Performing PCA analysis...")
        
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # Calculate explained variance
        explained_var_ratio = pca.explained_variance_ratio_
        cumulative_var_ratio = np.cumsum(explained_var_ratio)
        
        # Find components for 90% variance
        n_components_90 = np.argmax(cumulative_var_ratio >= 0.9) + 1
        
        self.models['pca'] = pca
        
        return {
            'n_components_90': n_components_90,
            'explained_variance_ratio': explained_var_ratio,
            'cumulative_variance_ratio': cumulative_var_ratio,
            'components': X_pca
        }
    
    def _optimize_kmeans(self, X_scaled: np.ndarray, max_k: int = None) -> int:
        """Find optimal number of clusters."""
        max_k = max_k or self.config.model.kmeans_max_k
        
        silhouette_scores = []
        K_range = range(2, min(max_k + 1, len(X_scaled) // 10))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=self.config.model.random_state, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            silhouette_scores.append(score)
        
        # Find optimal k
        optimal_k = K_range[np.argmax(silhouette_scores)]
        
        return optimal_k
    
    def _apply_kmeans(self, X_scaled: np.ndarray, full_data: pd.DataFrame) -> Dict[str, Any]:
        """Apply K-means clustering."""
        logger.debug("Applying K-means clustering...")
        
        # Find optimal k
        optimal_k = self._optimize_kmeans(X_scaled)
        logger.info(f"Optimal K-means clusters: {optimal_k}")
        
        # Apply K-means
        kmeans = KMeans(
            n_clusters=optimal_k,
            random_state=self.config.model.random_state,
            n_init=10
        )
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add to dataframe
        full_data['KMeans_Cluster'] = clusters
        
        # Identify suspicious clusters
        cluster_stats = full_data.groupby('KMeans_Cluster').agg({
            'Fuel_Diff': ['mean', 'min'],
            'Anomaly_Score': 'mean'
        })
        
        # Suspicious if high anomaly score or large fuel loss
        suspicious_clusters = []
        for cluster in range(optimal_k):
            cluster_data = full_data[full_data['KMeans_Cluster'] == cluster]
            if (cluster_data.get('Anomaly_Score', 0).mean() > 0.3 or
                cluster_data.get('Fuel_Diff', 0).mean() < -2):
                suspicious_clusters.append(cluster)
        
        self.models['kmeans'] = kmeans
        
        return {
            'optimal_k': optimal_k,
            'clusters': clusters,
            'suspicious_clusters': suspicious_clusters,
            'cluster_sizes': pd.Series(clusters).value_counts().to_dict()
        }
    
    def _apply_dbscan(self, X_scaled: np.ndarray, full_data: pd.DataFrame) -> Dict[str, Any]:
        """Apply DBSCAN clustering."""
        logger.debug("Applying DBSCAN clustering...")
        
        dbscan = DBSCAN(
            eps=self.config.model.dbscan_eps,
            min_samples=self.config.model.dbscan_min_samples
        )
        clusters = dbscan.fit_predict(X_scaled)
        
        # Add to dataframe
        full_data['DBSCAN_Cluster'] = clusters
        
        # Count outliers
        n_outliers = (clusters == -1).sum()
        outlier_rate = n_outliers / len(clusters)
        
        logger.info(f"DBSCAN outliers: {n_outliers} ({outlier_rate:.2%})")
        
        self.models['dbscan'] = dbscan
        
        return {
            'clusters': clusters,
            'n_outliers': n_outliers,
            'outlier_rate': outlier_rate,
            'n_clusters': len(set(clusters)) - (1 if -1 in clusters else 0)
        }
    
    def _apply_isolation_forest(self, X_scaled: np.ndarray, full_data: pd.DataFrame) -> Dict[str, Any]:
        """Apply Isolation Forest."""
        logger.debug("Applying Isolation Forest...")
        
        iso_forest = IsolationForest(
            contamination=self.config.model.isolation_forest_contamination,
            random_state=self.config.model.random_state
        )
        predictions = iso_forest.fit_predict(X_scaled)
        scores = iso_forest.score_samples(X_scaled)
        
        # Add to dataframe
        full_data['IsoForest_Anomaly'] = (predictions == -1).astype(int)
        full_data['IsoForest_Score'] = scores
        
        # Count anomalies
        n_anomalies = (predictions == -1).sum()
        anomaly_rate = n_anomalies / len(predictions)
        
        logger.info(f"Isolation Forest anomalies: {n_anomalies} ({anomaly_rate:.2%})")
        
        self.models['isolation_forest'] = iso_forest
        
        return {
            'predictions': predictions,
            'scores': scores,
            'n_anomalies': n_anomalies,
            'anomaly_rate': anomaly_rate
        }
    
    def _create_ensemble_score(self, full_data: pd.DataFrame) -> Dict[str, Any]:
        """Create ensemble anomaly score."""
        logger.debug("Creating ensemble anomaly score...")
        
        # Initialize ensemble score
        full_data['Ensemble_Score'] = 0
        
        # Weights for different methods
        weights = {
            'anomaly_score': 0.3,
            'dbscan': 0.25,
            'isolation_forest': 0.25,
            'kmeans': 0.2
        }
        
        # Add weighted contributions
        if 'Anomaly_Score' in full_data.columns:
            full_data['Ensemble_Score'] += full_data['Anomaly_Score'] * weights['anomaly_score']
        
        if 'DBSCAN_Cluster' in full_data.columns:
            dbscan_anomaly = (full_data['DBSCAN_Cluster'] == -1).astype(float)
            full_data['Ensemble_Score'] += dbscan_anomaly * weights['dbscan']
        
        if 'IsoForest_Anomaly' in full_data.columns:
            full_data['Ensemble_Score'] += full_data['IsoForest_Anomaly'] * weights['isolation_forest']
        
        if 'KMeans_Cluster' in full_data.columns and hasattr(self, 'suspicious_clusters'):
            kmeans_suspicious = full_data['KMeans_Cluster'].isin(
                self.results.get('kmeans', {}).get('suspicious_clusters', [])
            ).astype(float)
            full_data['Ensemble_Score'] += kmeans_suspicious * weights['kmeans']
        
        # Normalize score
        if full_data['Ensemble_Score'].max() > 0:
            full_data['Ensemble_Score'] = full_data['Ensemble_Score'] / full_data['Ensemble_Score'].max()
        
        # Create predictions based on threshold
        threshold = full_data['Ensemble_Score'].quantile(0.95)
        full_data['Predicted_Anomaly'] = (full_data['Ensemble_Score'] > threshold).astype(int)
        
        n_anomalies = full_data['Predicted_Anomaly'].sum()
        
        return {
            'threshold': threshold,
            'n_anomalies': n_anomalies,
            'anomaly_rate': n_anomalies / len(full_data)
        }
    
    def plot_results(self, X: pd.DataFrame, full_data: pd.DataFrame) -> None:
        """Plot unsupervised learning results."""
        if 'pca' not in self.results:
            logger.warning("No results to plot. Run fit_predict first.")
            return
        
        # Use PCA components for visualization
        X_pca = self.results['pca']['components'][:, :2]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Unsupervised Learning Results', fontsize=16)
        
        # Plot K-means
        if 'KMeans_Cluster' in full_data.columns:
            ax = axes[0, 0]
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                               c=full_data['KMeans_Cluster'], 
                               cmap='viridis', alpha=0.6, s=10)
            ax.set_title('K-means Clustering')
            ax.set_xlabel('First Principal Component')
            ax.set_ylabel('Second Principal Component')
            plt.colorbar(scatter, ax=ax)
        
        # Plot DBSCAN
        if 'DBSCAN_Cluster' in full_data.columns:
            ax = axes[0, 1]
            colors = ['red' if x == -1 else 'blue' for x in full_data['DBSCAN_Cluster']]
            ax.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.6, s=10)
            ax.set_title('DBSCAN (Red = Outliers)')
            ax.set_xlabel('First Principal Component')
            ax.set_ylabel('Second Principal Component')
        
        # Plot Isolation Forest
        if 'IsoForest_Anomaly' in full_data.columns:
            ax = axes[1, 0]
            colors = ['red' if x == 1 else 'blue' for x in full_data['IsoForest_Anomaly']]
            ax.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.6, s=10)
            ax.set_title('Isolation Forest (Red = Anomalies)')
            ax.set_xlabel('First Principal Component')
            ax.set_ylabel('Second Principal Component')
        
        # Plot Ensemble
        if 'Predicted_Anomaly' in full_data.columns:
            ax = axes[1, 1]
            colors = ['red' if x == 1 else 'blue' for x in full_data['Predicted_Anomaly']]
            sizes = [50 if x == 1 else 10 for x in full_data['Predicted_Anomaly']]
            ax.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, s=sizes, alpha=0.6)
            ax.set_title('Ensemble Predictions (Red = Anomalies)')
            ax.set_xlabel('First Principal Component')
            ax.set_ylabel('Second Principal Component')
        
        plt.tight_layout()
        plt.show()