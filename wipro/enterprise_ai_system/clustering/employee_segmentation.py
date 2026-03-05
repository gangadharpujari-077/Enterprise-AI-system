"""
Employee Clustering and Segmentation Module
Segment employees into meaningful groups using KMeans clustering
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmployeeSegmentation:
    """Segment employees into performance and risk groups"""
    
    CLUSTER_LABELS = {
        0: "High Performer",
        1: "Stable Worker",
        2: "Burnout Risk",
        3: "Performance Concern",
        4: "Development Focus"
    }
    
    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.scaler = StandardScaler()
        self.X_scaled = None
        self.cluster_labels = None
        self.cluster_centers = None
    
    def fit(self, X: np.ndarray) -> Dict:
        """
        Fit KMeans clustering
        
        Args:
            X: Feature matrix
        
        Returns:
            Dictionary with cluster statistics
        """
        # Scale features
        self.X_scaled = self.scaler.fit_transform(X)
        
        # Fit KMeans
        self.kmeans = KMeans(n_clusters=self.n_clusters, 
                            random_state=self.random_state,
                            n_init=10)
        self.cluster_labels = self.kmeans.fit_predict(self.X_scaled)
        self.cluster_centers = self.kmeans.cluster_centers_
        
        # Calculate statistics
        stats = {}
        for i in range(self.n_clusters):
            cluster_mask = self.cluster_labels == i
            cluster_size = np.sum(cluster_mask)
            stats[i] = {
                'size': cluster_size,
                'percentage': cluster_size / len(self.cluster_labels) * 100,
                'inertia': self.kmeans.inertia_,
                'label': self._get_cluster_label(i)
            }
        
        logger.info(f"KMeans clustering completed with {self.n_clusters} clusters")
        for cluster_id, stat in stats.items():
            logger.info(f"Cluster {cluster_id} ({stat['label']}): {stat['percentage']:.1f}%")
        
        return stats
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data
        
        Args:
            X: Feature matrix
        
        Returns:
            Array of cluster labels
        """
        if self.kmeans is None:
            raise ValueError("Model not fitted yet")
        
        X_scaled = self.scaler.transform(X)
        return self.kmeans.predict(X_scaled)
    
    def get_cluster_characteristics(self, X: pd.DataFrame) -> Dict:
        """
        Analyze characteristics of each cluster
        
        Args:
            X: Feature dataframe
        
        Returns:
            Dictionary with cluster characteristics
        """
        if self.kmeans is None:
            raise ValueError("Model not fitted yet")
        
        characteristics = {}
        
        for cluster_id in range(self.n_clusters):
            mask = self.cluster_labels == cluster_id
            cluster_data = X[mask]
            
            characteristics[cluster_id] = {
                'label': self._get_cluster_label(cluster_id),
                'size': len(cluster_data),
                'feature_means': cluster_data.mean().to_dict(),
                'feature_stds': cluster_data.std().to_dict()
            }
        
        return characteristics
    
    def _get_cluster_label(self, cluster_id: int) -> str:
        """Get readable label for cluster"""
        if cluster_id < len(self.CLUSTER_LABELS):
            return self.CLUSTER_LABELS[cluster_id]
        return f"Cluster {cluster_id}"
    
    def get_optimal_clusters(self, X: np.ndarray, 
                           k_range: Tuple = (2, 10)) -> int:
        """
        Find optimal number of clusters using elbow method
        
        Args:
            X: Feature matrix
            k_range: Range of k values to test
        
        Returns:
            Optimal number of clusters
        """
        inertias = []
        k_values = range(k_range[0], k_range[1] + 1)
        
        X_scaled = self.scaler.fit_transform(X)
        
        for k in k_values:
            kmeans_temp = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans_temp.fit(X_scaled)
            inertias.append(kmeans_temp.inertia_)
        
        # Simple elbow detection: find biggest drop
        differences = np.diff(inertias)
        optimal_k = k_values[np.argmax(np.diff(differences)) + 1]
        
        logger.info(f"Optimal clusters: {optimal_k}")
        return optimal_k
    
    def visualize_clusters_pca(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project clusters to 2D using PCA for visualization
        
        Args:
            X: Feature matrix
        
        Returns:
            Tuple of (PCA components, labels)
        """
        if self.kmeans is None:
            raise ValueError("Model not fitted yet")
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_scaled)
        
        logger.info(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
        
        return X_pca, self.cluster_labels


class EmployeeRiskProfile:
    """Generate risk profiles based on clustering and metrics"""
    
    RISK_THRESHOLDS = {
        'high_performer': {'delay_risk': 0.2, 'burnout_risk': 0.1},
        'stable_worker': {'delay_risk': 0.4, 'burnout_risk': 0.3},
        'burnout_risk': {'delay_risk': 0.6, 'burnout_risk': 0.7},
        'performance_concern': {'delay_risk': 0.8, 'burnout_risk': 0.6}
    }
    
    @staticmethod
    def assign_risk_level(delay_risk: float, 
                         burnout_risk: float) -> str:
        """
        Assign risk level based on risk scores
        
        Args:
            delay_risk: Delay risk probability (0-1)
            burnout_risk: Burnout risk probability (0-1)
        
        Returns:
            Risk level: 'low', 'medium', 'high', 'critical'
        """
        combined_risk = (delay_risk + burnout_risk) / 2
        
        if combined_risk < 0.2:
            return 'low'
        elif combined_risk < 0.4:
            return 'medium'
        elif combined_risk < 0.7:
            return 'high'
        else:
            return 'critical'
    
    @staticmethod
    def generate_recommendations(cluster_label: str,
                                delay_risk: float,
                                burnout_risk: float,
                                cluster_characteristics: Dict) -> List[str]:
        """
        Generate recommendations based on risk profile
        
        Args:
            cluster_label: Employee cluster label
            delay_risk: Delay risk probability
            burnout_risk: Burnout risk probability
            cluster_characteristics: Cluster analysis results
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Delay risk recommendations
        if delay_risk > 0.5:
            recommendations.append("Priority: Review pending tasks and deadlines")
            recommendations.append("Action: Prioritize urgent tasks and reduce non-critical workload")
        
        # Burnout risk recommendations
        if burnout_risk > 0.5:
            recommendations.append("Alert: High burnout risk detected")
            recommendations.append("Action: Schedule wellness check-in and consider workload reduction")
            recommendations.append("Action: Monitor overtime hours closely")
        
        # Cluster-specific recommendations
        if 'Burnout Risk' in cluster_label:
            recommendations.append("Cluster insight: This employee is in high-risk group")
            recommendations.append("Action: Increase check-in frequency")
        elif 'High Performer' in cluster_label:
            recommendations.append("Strength: Maintaining high performance levels")
            recommendations.append("Action: Consider for leadership opportunities")
        
        # Workload recommendations
        if burnout_risk > 0.4 or delay_risk > 0.5:
            recommendations.append("Action: Redistribute workload or extend deadlines")
        
        return recommendations
