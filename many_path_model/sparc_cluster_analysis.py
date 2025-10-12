#!/usr/bin/env python3
"""
SPARC Galaxy Clustering Analysis
=================================

Analyzes optimization results to cluster galaxies based on their parameter sensitivity
and response patterns. Uses multiple clustering algorithms to find galaxies that behave
similarly under parameter variations.

Features:
- Feature engineering from optimization results
- Parameter sensitivity analysis
- Multiple clustering algorithms (K-Means, DBSCAN, Hierarchical)
- PCA dimensionality reduction
- Interactive visualizations
- Cluster validation metrics

Usage:
    # Analyze mega parallel results with K-Means
    python many_path_model/sparc_cluster_analysis.py --results results/mega_parallel/mega_parallel_results.json --output_dir results/clustering --method kmeans --n_clusters 5

    # Try all clustering methods
    python many_path_model/sparc_cluster_analysis.py --results results/mega_parallel/mega_parallel_results.json --output_dir results/clustering --method all

    # DBSCAN for automatic cluster detection
    python many_path_model/sparc_cluster_analysis.py --results results/mega_parallel/mega_parallel_results.json --output_dir results/clustering --method dbscan
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# Clustering and analysis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Visualization
import matplotlib.pyplot as plt


def extract_features(results: List[Dict]) -> pd.DataFrame:
    """
    Extract meaningful features from optimization results.
    
    Creates a feature vector for each galaxy capturing:
    - Best error achieved
    - Parameter values at best solution
    - Variation across restarts (robustness)
    - Convergence behavior
    """
    features = []
    
    for galaxy in results:
        if not galaxy['success']:
            continue
        
        feat = {
            'name': galaxy['name'],
            'hubble_type': galaxy['hubble_type'],
            'type_group': galaxy['type_group'],
            
            # Performance metrics
            'best_error': galaxy['best_error'],
            'total_evaluations': galaxy['total_evaluations'],
            
            # Best parameters
            'eta': galaxy['best_params']['eta'],
            'ring_amp': galaxy['best_params']['ring_amp'],
            'M_max': galaxy['best_params']['M_max'],
            'bulge_gate_power': galaxy['best_params']['bulge_gate_power'],
            'lambda_hat': galaxy['best_params']['lambda_hat'],
        }
        
        # Analyze all attempts to get variation/robustness
        if 'all_attempts' in galaxy and len(galaxy['all_attempts']) > 1:
            errors = [a['error'] for a in galaxy['all_attempts']]
            
            feat['error_std'] = np.std(errors)
            feat['error_range'] = np.max(errors) - np.min(errors)
            feat['error_cv'] = np.std(errors) / np.mean(errors) if np.mean(errors) > 0 else 0
            
            # Parameter stability across restarts
            param_arrays = {
                'eta': [a['params']['eta'] for a in galaxy['all_attempts']],
                'ring_amp': [a['params']['ring_amp'] for a in galaxy['all_attempts']],
                'M_max': [a['params']['M_max'] for a in galaxy['all_attempts']],
                'bulge_gate_power': [a['params']['bulge_gate_power'] for a in galaxy['all_attempts']],
                'lambda_hat': [a['params']['lambda_hat'] for a in galaxy['all_attempts']],
            }
            
            for param_name, values in param_arrays.items():
                feat[f'{param_name}_std'] = np.std(values)
                feat[f'{param_name}_range'] = np.max(values) - np.min(values)
        else:
            # Single attempt - no variation metrics
            feat['error_std'] = 0
            feat['error_range'] = 0
            feat['error_cv'] = 0
            for param_name in ['eta', 'ring_amp', 'M_max', 'bulge_gate_power', 'lambda_hat']:
                feat[f'{param_name}_std'] = 0
                feat[f'{param_name}_range'] = 0
        
        features.append(feat)
    
    return pd.DataFrame(features)


def prepare_clustering_data(df: pd.DataFrame) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Prepare feature matrix for clustering.
    Returns: (feature_matrix, feature_names, metadata_df)
    """
    # Metadata to keep separate
    metadata_cols = ['name', 'hubble_type', 'type_group']
    metadata = df[metadata_cols].copy()
    
    # Features for clustering (exclude metadata)
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    X = df[feature_cols].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, feature_cols, metadata


def perform_pca(X: np.ndarray, n_components: int = 10) -> Tuple[np.ndarray, PCA]:
    """Dimensionality reduction with PCA."""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    print(f"\nPCA Results:")
    print(f"  Components: {n_components}")
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    print(f"  Top 3 components: {pca.explained_variance_ratio_[:3]}")
    
    return X_pca, pca


def cluster_kmeans(X: np.ndarray, n_clusters: int) -> np.ndarray:
    """K-Means clustering."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X)
    return labels


def cluster_dbscan(X: np.ndarray, eps: float = 0.5, min_samples: int = 3) -> np.ndarray:
    """DBSCAN clustering."""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    return labels


def cluster_hierarchical(X: np.ndarray, n_clusters: int) -> np.ndarray:
    """Hierarchical/Agglomerative clustering."""
    agglo = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = agglo.fit_predict(X)
    return labels


def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> Dict:
    """Compute clustering quality metrics."""
    # Filter out noise points (label = -1 from DBSCAN)
    mask = labels != -1
    X_filtered = X[mask]
    labels_filtered = labels[mask]
    
    n_clusters = len(set(labels_filtered))
    n_noise = np.sum(labels == -1)
    
    metrics = {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
    }
    
    # Can only compute these if we have multiple clusters
    if n_clusters > 1 and len(X_filtered) > n_clusters:
        try:
            metrics['silhouette'] = silhouette_score(X_filtered, labels_filtered)
            metrics['calinski_harabasz'] = calinski_harabasz_score(X_filtered, labels_filtered)
            metrics['davies_bouldin'] = davies_bouldin_score(X_filtered, labels_filtered)
        except:
            pass
    
    return metrics


def plot_pca_scatter(X_pca: np.ndarray, labels: np.ndarray, metadata: pd.DataFrame, 
                     output_dir: Path, method_name: str):
    """Plot 2D PCA scatter colored by cluster."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Colored by cluster
    scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', 
                               alpha=0.6, s=50)
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].set_title(f'Clusters - {method_name}')
    plt.colorbar(scatter1, ax=axes[0], label='Cluster')
    
    # Plot 2: Colored by morphological type
    type_map = {'early': 0, 'intermediate': 1, 'late': 2, 'unknown': 3}
    type_colors = metadata['type_group'].map(type_map)
    scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=type_colors, cmap='viridis',
                               alpha=0.6, s=50)
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].set_title('Morphological Type')
    cbar = plt.colorbar(scatter2, ax=axes[1], ticks=[0, 1, 2, 3])
    cbar.set_ticklabels(['Early', 'Intermediate', 'Late', 'Unknown'])
    
    plt.tight_layout()
    plt.savefig(output_dir / f'pca_scatter_{method_name}.png', dpi=150)
    plt.close()


def plot_cluster_characteristics(df_clustered: pd.DataFrame, output_dir: Path, method_name: str):
    """Plot characteristics of each cluster."""
    n_clusters = df_clustered['cluster'].nunique()
    
    # Key parameters to visualize
    param_cols = ['best_error', 'eta', 'ring_amp', 'M_max', 'bulge_gate_power', 'lambda_hat']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, param in enumerate(param_cols):
        if param in df_clustered.columns:
            df_clustered.boxplot(column=param, by='cluster', ax=axes[i])
            axes[i].set_title(param)
            axes[i].set_xlabel('Cluster')
    
    plt.suptitle(f'Cluster Characteristics - {method_name}', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / f'cluster_characteristics_{method_name}.png', dpi=150)
    plt.close()


def analyze_cluster_composition(df_clustered: pd.DataFrame, output_dir: Path, method_name: str):
    """Analyze and save cluster composition."""
    summary = []
    
    for cluster_id in sorted(df_clustered['cluster'].unique()):
        cluster_df = df_clustered[df_clustered['cluster'] == cluster_id]
        
        summary.append({
            'cluster': cluster_id,
            'size': len(cluster_df),
            'mean_error': cluster_df['best_error'].mean(),
            'std_error': cluster_df['best_error'].std(),
            'early_count': (cluster_df['type_group'] == 'early').sum(),
            'intermediate_count': (cluster_df['type_group'] == 'intermediate').sum(),
            'late_count': (cluster_df['type_group'] == 'late').sum(),
            'mean_eta': cluster_df['eta'].mean(),
            'mean_ring_amp': cluster_df['ring_amp'].mean(),
            'mean_M_max': cluster_df['M_max'].mean(),
            'mean_bulge_gate_power': cluster_df['bulge_gate_power'].mean(),
            'mean_lambda_hat': cluster_df['lambda_hat'].mean(),
        })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_dir / f'cluster_summary_{method_name}.csv', index=False)
    
    return summary_df


def main():
    parser = argparse.ArgumentParser(description='SPARC Galaxy Clustering Analysis')
    parser.add_argument('--results', required=True, help='Path to mega parallel results JSON')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--method', default='kmeans', choices=['kmeans', 'dbscan', 'hierarchical', 'all'],
                       help='Clustering method')
    parser.add_argument('--n_clusters', type=int, default=5, help='Number of clusters (for kmeans/hierarchical)')
    parser.add_argument('--pca_components', type=int, default=10, help='Number of PCA components')
    parser.add_argument('--dbscan_eps', type=float, default=0.8, help='DBSCAN epsilon parameter')
    parser.add_argument('--dbscan_min_samples', type=int, default=3, help='DBSCAN min_samples parameter')
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print(f"Loading results from {args.results}...")
    with open(args.results, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    successful = [r for r in results if r['success']]
    
    print(f"Loaded {len(successful)} successful galaxy optimizations")
    
    # Extract features
    print("\nExtracting features...")
    df = extract_features(successful)
    print(f"Created {len(df)} feature vectors with {len(df.columns)} columns")
    
    # Save feature matrix
    df.to_csv(output_dir / 'features.csv', index=False)
    print(f"Saved features to: {output_dir / 'features.csv'}")
    
    # Prepare for clustering
    X_scaled, feature_names, metadata = prepare_clustering_data(df)
    print(f"\nClustering features: {len(feature_names)}")
    
    # Apply PCA
    X_pca, pca = perform_pca(X_scaled, n_components=args.pca_components)
    
    # Plot PCA variance
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title('PCA Explained Variance')
    plt.tight_layout()
    plt.savefig(output_dir / 'pca_variance.png', dpi=150)
    plt.close()
    
    # Clustering methods
    methods_to_run = []
    if args.method == 'all':
        methods_to_run = ['kmeans', 'dbscan', 'hierarchical']
    else:
        methods_to_run = [args.method]
    
    print("\n" + "=" * 80)
    print("CLUSTERING ANALYSIS")
    print("=" * 80)
    
    for method in methods_to_run:
        print(f"\n{method.upper()} Clustering:")
        print("-" * 40)
        
        # Apply clustering
        if method == 'kmeans':
            labels = cluster_kmeans(X_pca, args.n_clusters)
        elif method == 'dbscan':
            labels = cluster_dbscan(X_pca, eps=args.dbscan_eps, min_samples=args.dbscan_min_samples)
        elif method == 'hierarchical':
            labels = cluster_hierarchical(X_pca, args.n_clusters)
        
        # Evaluate
        metrics = evaluate_clustering(X_pca, labels)
        print(f"Number of clusters: {metrics['n_clusters']}")
        print(f"Noise points: {metrics.get('n_noise', 0)}")
        if 'silhouette' in metrics:
            print(f"Silhouette Score: {metrics['silhouette']:.3f}")
            print(f"Calinski-Harabasz: {metrics['calinski_harabasz']:.1f}")
            print(f"Davies-Bouldin: {metrics['davies_bouldin']:.3f}")
        
        # Add cluster labels to dataframe
        df_clustered = df.copy()
        df_clustered['cluster'] = labels
        df_clustered['pc1'] = X_pca[:, 0]
        df_clustered['pc2'] = X_pca[:, 1]
        
        # Save clustered data
        df_clustered.to_csv(output_dir / f'clustered_{method}.csv', index=False)
        
        # Visualizations
        plot_pca_scatter(X_pca, labels, metadata, output_dir, method)
        plot_cluster_characteristics(df_clustered, output_dir, method)
        summary_df = analyze_cluster_composition(df_clustered, output_dir, method)
        
        print(f"\nCluster Summary:")
        print(summary_df.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
