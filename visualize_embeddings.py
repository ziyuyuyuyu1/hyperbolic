#!/usr/bin/env python3
"""
Token Embedding Visualization Script

This script provides various methods to visualize token embeddings from language models,
with special support for hyperbolic embeddings (Poincaré and Lorentz models).

Usage:
    python visualize_embeddings.py --model_path <path> --output_dir <dir> [options]
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from llamafactory.model.hyperbolic_utils import (
    lift_to_lorentz, 
    project_to_poincare_ball, 
    poincare_exp_map, 
    lorentz_exp_map,
    poincare_distance,
    lorentz_distance
)

class EmbeddingVisualizer:
    """Class for visualizing token embeddings with support for hyperbolic spaces."""
    
    def __init__(self, model_path: str, device: str = 'cpu', raw_embeddings: bool = False):
        """
        Initialize the visualizer with a model.
        
        Args:
            model_path: Path to the model directory
            device: Device to load the model on ('cpu' or 'cuda')
        """
        self.model_path = model_path
        self.device = device
        self.raw_embeddings = raw_embeddings
        self.model = None
        self.tokenizer = None
        self.embeddings = None
        self.embedding_space = None  # 'euclidean', 'poincare', or 'lorentz'
        
        self._load_model()
        self._extract_embeddings()
    
    def _load_model(self):
        """Load the model and tokenizer."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print(f"Loading model from {self.model_path}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, 
                torch_dtype=torch.float32,
                device_map='auto' if self.device == 'cuda' else None
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Determine embedding space type
            model_type = self.model.config.model_type if hasattr(self.model.config, 'model_type') else 'unknown'
            if 'poincare' in model_type.lower():
                self.embedding_space = 'poincare'
            elif 'lorentz' in model_type.lower():
                self.embedding_space = 'lorentz'
            else:
                self.embedding_space = 'euclidean'
                
            print(f"Model loaded successfully. Embedding space: {self.embedding_space}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _extract_embeddings(self):
        """Extract token embeddings from the model."""
        print("Extracting token embeddings...")
        
        # Get the embedding layer
        if hasattr(self.model, 'get_input_embeddings'):
            embedding_layer = self.model.get_input_embeddings()
        else:
            embedding_layer = self.model.model.embed_tokens
            
        # Extract raw embeddings
        raw_embeddings = embedding_layer.weight.detach().cpu().numpy()
        
        # Transform embeddings based on space type
        if self.embedding_space == 'poincare':
            # For Poincaré models, we need to apply exponential map
            raw_tensor = torch.tensor(raw_embeddings, dtype=torch.float32)
            if self.raw_embeddings:
                self.embeddings = raw_tensor.numpy()
            else:
                self.embeddings = poincare_exp_map(raw_tensor).numpy()
                # Project to Poincaré ball
                self.embeddings = project_to_poincare_ball(
                    torch.tensor(self.embeddings, dtype=torch.float32)
                ).numpy()
            
        elif self.embedding_space == 'lorentz':
            # For Lorentz models, apply exponential map
            raw_tensor = torch.tensor(raw_embeddings, dtype=torch.float32)
            if self.raw_embeddings:
                self.embeddings = raw_tensor.numpy()
            else:
                self.embeddings = lorentz_exp_map(raw_tensor).numpy()
            
        else:
            # Euclidean space - use raw embeddings
            self.embeddings = raw_embeddings
            
        print(f"Extracted {self.embeddings.shape[0]} token embeddings of dimension {self.embeddings.shape[1]}")
    
    def get_token_info(self, token_ids: List[int]) -> Dict[str, List]:
        """Get information about tokens."""
        info = {
            'token_id': [],
            'token_text': [],
            'token_repr': [],
            'category': []
        }
        
        for token_id in token_ids:
            token_text = self.tokenizer.decode([token_id])
            info['token_id'].append(token_id)
            info['token_text'].append(token_text)
            info['token_repr'].append(repr(token_text))
            
            # Categorize tokens
            if token_text.isspace():
                category = 'whitespace'
            elif token_text.isalpha():
                category = 'alphabetic'
            elif token_text.isdigit():
                category = 'numeric'
            elif token_text in '.,;:!?()[]{}"\'':
                category = 'punctuation'
            elif token_text.startswith('Ġ'):  # BPE token prefix
                category = 'word_start'
            else:
                category = 'other'
            
            info['category'].append(category)
        
        return info
    
    def sample_tokens(self, n_samples: int = 1000, random_seed: int = 42) -> Tuple[np.ndarray, Dict]:
        """Sample tokens for visualization."""
        np.random.seed(random_seed)
        
        vocab_size = len(self.tokenizer)
        if n_samples >= vocab_size:
            # Use all tokens
            sampled_indices = np.arange(vocab_size)
        else:
            # Sample tokens
            sampled_indices = np.random.choice(vocab_size, n_samples, replace=False)
        
        sampled_embeddings = self.embeddings[sampled_indices]
        token_info = self.get_token_info(sampled_indices.tolist())
        
        return sampled_embeddings, token_info
    
    def compute_distances(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between embeddings."""
        if self.embedding_space == 'poincare':
            # Use Poincaré distance
            embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
            distances = poincare_distance(
                embeddings_tensor.unsqueeze(1),
                embeddings_tensor.unsqueeze(0)
            ).numpy()
        elif self.embedding_space == 'lorentz':
            # Use Lorentz distance
            embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
            distances = lorentz_distance(
                embeddings_tensor,
                embeddings_tensor
            ).numpy()
        else:
            # Use Euclidean distance
            distances = np.linalg.norm(embeddings[:, np.newaxis] - embeddings, axis=2)
        
        return distances
    
    def reduce_dimensions(self, embeddings: np.ndarray, method: str = 'tsne', 
                         n_components: int = 2, random_state: int = 42) -> np.ndarray:
        """Reduce embedding dimensions for visualization."""
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=random_state, 
                          perplexity=min(30, len(embeddings) - 1))
        elif method.lower() == 'pca':
            reducer = PCA(n_components=n_components, random_state=random_state)
        else:
            raise ValueError(f"Unknown reduction method: {method}")
        
        return reducer.fit_transform(embeddings)
    
    def plot_embeddings_2d(self, embeddings_2d: np.ndarray, token_info: Dict, 
                          output_path: str, title: str = "Token Embeddings"):
        """Create 2D scatter plot of embeddings."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Color by category
        categories = token_info['category']
        unique_categories = list(set(categories))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_categories)))
        color_map = dict(zip(unique_categories, colors))
        
        # Plot 1: Colored by category
        for category in unique_categories:
            mask = [c == category for c in categories]
            ax1.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       c=[color_map[category]], label=category, alpha=0.7, s=20)
        
        ax1.set_title(f"{title} - By Category")
        ax1.set_xlabel("Component 1")
        ax1.set_ylabel("Component 2")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Colored by token ID (to show clustering)
        scatter = ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                             c=token_info['token_id'], cmap='viridis', alpha=0.7, s=20)
        ax2.set_title(f"{title} - By Token ID")
        ax2.set_xlabel("Component 1")
        ax2.set_ylabel("Component 2")
        plt.colorbar(scatter, ax=ax2, label='Token ID')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"2D plot saved to {output_path}")
    
    def plot_embeddings_2d_with_labels(self, embeddings_2d: np.ndarray, token_info: Dict, 
                                      output_path: str, title: str = "Token Embeddings with Labels",
                                      max_labels: int = 100):
        """Create 2D scatter plot with token labels and clustering."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
        
        # Color by category
        categories = token_info['category']
        unique_categories = list(set(categories))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_categories)))
        color_map = dict(zip(unique_categories, colors))
        
        # Plot 1: Colored by category with labels
        for category in unique_categories:
            mask = [c == category for c in categories]
            category_points = embeddings_2d[mask]
            category_tokens = [token_info['token_text'][i] for i, m in enumerate(mask) if m]
            category_ids = [token_info['token_id'][i] for i, m in enumerate(mask) if m]
            
            ax1.scatter(category_points[:, 0], category_points[:, 1], 
                       c=[color_map[category]], label=category, alpha=0.7, s=30)
            
            # Add labels for some tokens in each category
            if len(category_points) <= max_labels:
                # Label all tokens in small categories
                for i, (point, token, token_id) in enumerate(zip(category_points, category_tokens, category_ids)):
                    # Clean up token text for display
                    display_text = token.replace('\n', '\\n').replace('\t', '\\t')
                    if len(display_text) > 15:
                        display_text = display_text[:12] + '...'
                    ax1.annotate(f"{token_id}:{display_text}", 
                               (point[0], point[1]), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=6, alpha=0.8)
            else:
                # Sample tokens for labeling in large categories
                sample_indices = np.random.choice(len(category_points), max_labels//len(unique_categories), replace=False)
                for idx in sample_indices:
                    point = category_points[idx]
                    token = category_tokens[idx]
                    token_id = category_ids[idx]
                    display_text = token.replace('\n', '\\n').replace('\t', '\\t')
                    if len(display_text) > 15:
                        display_text = display_text[:12] + '...'
                    ax1.annotate(f"{token_id}:{display_text}", 
                               (point[0], point[1]), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=6, alpha=0.8)
        
        ax1.set_title(f"{title} - By Category (with Token Labels)")
        ax1.set_xlabel("Component 1")
        ax1.set_ylabel("Component 2")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Clustering with K-means and cluster boundaries
        from sklearn.cluster import KMeans
        from scipy.spatial import ConvexHull
        
        # Perform clustering
        n_clusters = min(8, len(embeddings_2d) // 10)  # Adaptive number of clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_2d)
        
        # Plot clusters with different colors
        cluster_colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_points = embeddings_2d[cluster_mask]
            cluster_tokens = [token_info['token_text'][i] for i, m in enumerate(cluster_mask) if m]
            cluster_ids = [token_info['token_id'][i] for i, m in enumerate(cluster_mask) if m]
            
            ax2.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       c=[cluster_colors[cluster_id]], label=f'Cluster {cluster_id}', 
                       alpha=0.7, s=30)
            
            # Draw cluster boundary (convex hull)
            if len(cluster_points) > 3:
                try:
                    hull = ConvexHull(cluster_points)
                    for simplex in hull.simplices:
                        ax2.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], 
                                'k-', alpha=0.3, linewidth=1)
                except:
                    pass  # Skip if convex hull fails
            
            # Add cluster center
            center = kmeans.cluster_centers_[cluster_id]
            ax2.scatter(center[0], center[1], c='red', s=100, marker='x', linewidths=3, 
                       label=f'Center {cluster_id}' if cluster_id == 0 else "")
            
            # Enhanced labeling for clustering - show more prominent token names
            if len(cluster_points) <= max_labels:
                # Label all tokens in small clusters
                for i, (point, token, token_id) in enumerate(zip(cluster_points, cluster_tokens, cluster_ids)):
                    display_text = token.replace('\n', '\\n').replace('\t', '\\t')
                    if len(display_text) > 20:
                        display_text = display_text[:17] + '...'
                    
                    # Use larger font and better positioning for clustering view
                    ax2.annotate(f"{display_text}", 
                               (point[0], point[1]), 
                               xytext=(8, 8), textcoords='offset points',
                               fontsize=8, alpha=0.9, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
                    
                    # Add token ID as smaller text
                    ax2.annotate(f"ID:{token_id}", 
                               (point[0], point[1]), 
                               xytext=(8, -8), textcoords='offset points',
                               fontsize=6, alpha=0.7, color='red')
            else:
                # Sample tokens for labeling in large clusters
                sample_indices = np.random.choice(len(cluster_points), max_labels//n_clusters, replace=False)
                for idx in sample_indices:
                    point = cluster_points[idx]
                    token = cluster_tokens[idx]
                    token_id = cluster_ids[idx]
                    display_text = token.replace('\n', '\\n').replace('\t', '\\t')
                    if len(display_text) > 20:
                        display_text = display_text[:17] + '...'
                    
                    # Use larger font and better positioning for clustering view
                    ax2.annotate(f"{display_text}", 
                               (point[0], point[1]), 
                               xytext=(8, 8), textcoords='offset points',
                               fontsize=8, alpha=0.9, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
                    
                    # Add token ID as smaller text
                    ax2.annotate(f"ID:{token_id}", 
                               (point[0], point[1]), 
                               xytext=(8, -8), textcoords='offset points',
                               fontsize=6, alpha=0.7, color='red')
        
        ax2.set_title(f"{title} - Clustering Analysis ({n_clusters} clusters)")
        ax2.set_xlabel("Component 1")
        ax2.set_ylabel("Component 2")
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"2D plot with labels saved to {output_path}")
        
        return cluster_labels
    
    def plot_clustering_with_token_names(self, embeddings_2d: np.ndarray, token_info: Dict, 
                                        output_path: str, title: str = "Token Clustering with Names",
                                        max_labels_per_cluster: int = 20):
        """Create a clean clustering visualization with prominent token names."""
        from sklearn.cluster import KMeans
        from scipy.spatial import ConvexHull
        
        # Perform clustering
        n_clusters = min(6, len(embeddings_2d) // 15)  # Fewer clusters for better visibility
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_2d)
        
        # Create figure with larger size for better readability
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        # Plot clusters with different colors
        cluster_colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
        
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_points = embeddings_2d[cluster_mask]
            cluster_tokens = [token_info['token_text'][i] for i, m in enumerate(cluster_mask) if m]
            cluster_ids = [token_info['token_id'][i] for i, m in enumerate(cluster_mask) if m]
            cluster_categories = [token_info['category'][i] for i, m in enumerate(cluster_mask) if m]
            
            # Plot cluster points
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                      c=[cluster_colors[cluster_id]], label=f'Cluster {cluster_id}', 
                      alpha=0.6, s=40)
            
            # Draw cluster boundary
            if len(cluster_points) > 3:
                try:
                    hull = ConvexHull(cluster_points)
                    for simplex in hull.simplices:
                        ax.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], 
                               'k-', alpha=0.4, linewidth=2)
                except:
                    pass
            
            # Add cluster center
            center = kmeans.cluster_centers_[cluster_id]
            ax.scatter(center[0], center[1], c='red', s=150, marker='x', linewidths=4, 
                      label=f'Center {cluster_id}' if cluster_id == 0 else "")
            
            # Add token names with better positioning
            if len(cluster_points) <= max_labels_per_cluster:
                # Label all tokens in small clusters
                for i, (point, token, token_id, category) in enumerate(zip(cluster_points, cluster_tokens, cluster_ids, cluster_categories)):
                    # Clean up token text
                    display_text = token.replace('\n', '\\n').replace('\t', '\\t')
                    if len(display_text) > 25:
                        display_text = display_text[:22] + '...'
                    
                    # Color code by category
                    if category == 'alphabetic':
                        text_color = 'blue'
                    elif category == 'numeric':
                        text_color = 'red'
                    elif category == 'punctuation':
                        text_color = 'green'
                    elif category == 'whitespace':
                        text_color = 'gray'
                    else:
                        text_color = 'black'
                    
                    # Add token name with prominent styling
                    ax.annotate(display_text, 
                               (point[0], point[1]), 
                               xytext=(10, 10), textcoords='offset points',
                               fontsize=9, alpha=0.95, fontweight='bold', color=text_color,
                               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='black', linewidth=0.5))
                    
                    # Add token ID as smaller text below
                    ax.annotate(f"ID:{token_id}", 
                               (point[0], point[1]), 
                               xytext=(10, -5), textcoords='offset points',
                               fontsize=7, alpha=0.8, color='darkred')
            else:
                # Sample tokens for labeling in large clusters
                sample_indices = np.random.choice(len(cluster_points), max_labels_per_cluster, replace=False)
                for idx in sample_indices:
                    point = cluster_points[idx]
                    token = cluster_tokens[idx]
                    token_id = cluster_ids[idx]
                    category = cluster_categories[idx]
                    
                    display_text = token.replace('\n', '\\n').replace('\t', '\\t')
                    if len(display_text) > 25:
                        display_text = display_text[:22] + '...'
                    
                    # Color code by category
                    if category == 'alphabetic':
                        text_color = 'blue'
                    elif category == 'numeric':
                        text_color = 'red'
                    elif category == 'punctuation':
                        text_color = 'green'
                    elif category == 'whitespace':
                        text_color = 'gray'
                    else:
                        text_color = 'black'
                    
                    # Add token name with prominent styling
                    ax.annotate(display_text, 
                               (point[0], point[1]), 
                               xytext=(10, 10), textcoords='offset points',
                               fontsize=9, alpha=0.95, fontweight='bold', color=text_color,
                               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='black', linewidth=0.5))
                    
                    # Add token ID as smaller text below
                    ax.annotate(f"ID:{token_id}", 
                               (point[0], point[1]), 
                               xytext=(10, -5), textcoords='offset points',
                               fontsize=7, alpha=0.8, color='darkred')
        
        ax.set_title(f"{title} - {n_clusters} Clusters", fontsize=16, fontweight='bold')
        ax.set_xlabel("Component 1", fontsize=12)
        ax.set_ylabel("Component 2", fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Clustering with token names saved to {output_path}")
        
        return cluster_labels
    
    def plot_mathematical_tokens(self, embeddings_2d: np.ndarray, token_info: Dict, 
                                output_path: str, title: str = "Mathematical Token Embeddings"):
        """Specialized visualization for mathematical tokens (special symbols, operations, numbers)."""
        from sklearn.cluster import KMeans
        from scipy.spatial import ConvexHull
        
        # Perform clustering
        n_clusters = min(8, len(embeddings_2d) // 10)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_2d)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        # Define special tokens and operations
        special_tokens = ['+', '-', '*', '/', '=', '(', ')', '[', ']', '{', '}', ',', '.', ';', ':', '!', '?']
        operation_tokens = ['+', '-', '*', '/', '=', '^', '**', '//', '%', '==', '!=', '<=', '>=', '<', '>']
        
        # Plot clusters with different colors
        cluster_colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
        
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_points = embeddings_2d[cluster_mask]
            cluster_tokens = [token_info['token_text'][i] for i, m in enumerate(cluster_mask) if m]
            cluster_ids = [token_info['token_id'][i] for i, m in enumerate(cluster_mask) if m]
            
            # Plot cluster points
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                      c=[cluster_colors[cluster_id]], label=f'Cluster {cluster_id}', 
                      alpha=0.6, s=40)
            
            # Draw cluster boundary
            if len(cluster_points) > 3:
                try:
                    hull = ConvexHull(cluster_points)
                    for simplex in hull.simplices:
                        ax.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], 
                               'k-', alpha=0.3, linewidth=1)
                except:
                    pass
            
            # Add cluster center
            center = kmeans.cluster_centers_[cluster_id]
            ax.scatter(center[0], center[1], c='red', s=150, marker='x', linewidths=4, 
                      label=f'Center {cluster_id}' if cluster_id == 0 else "")
            
            # Categorize tokens in this cluster
            special_in_cluster = []
            operations_in_cluster = []
            numbers_in_cluster = []
            
            for i, (point, token, token_id) in enumerate(zip(cluster_points, cluster_tokens, cluster_ids)):
                # Check if it's a special token
                if token in special_tokens:
                    special_in_cluster.append((point, token, token_id))
                # Check if it's an operation
                elif token in operation_tokens:
                    operations_in_cluster.append((point, token, token_id))
                # Check if it's a number
                elif token.isdigit():
                    numbers_in_cluster.append((point, token, token_id))
            
            # Label special tokens prominently
            for point, token, token_id in special_in_cluster:
                ax.annotate(f"{token}", 
                           (point[0], point[1]), 
                           xytext=(8, 8), textcoords='offset points',
                           fontsize=12, alpha=0.95, fontweight='bold', color='red',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8, edgecolor='red', linewidth=2))
                ax.annotate(f"ID:{token_id}", 
                           (point[0], point[1]), 
                           xytext=(8, -8), textcoords='offset points',
                           fontsize=8, alpha=0.8, color='darkred')
            
            # Label operation tokens prominently
            for point, token, token_id in operations_in_cluster:
                if token not in special_tokens:  # Avoid double labeling
                    ax.annotate(f"{token}", 
                               (point[0], point[1]), 
                               xytext=(8, 8), textcoords='offset points',
                               fontsize=11, alpha=0.95, fontweight='bold', color='blue',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8, edgecolor='blue', linewidth=2))
                    ax.annotate(f"ID:{token_id}", 
                               (point[0], point[1]), 
                               xytext=(8, -8), textcoords='offset points',
                               fontsize=8, alpha=0.8, color='darkblue')
            
            # Handle number ranges
            if numbers_in_cluster:
                # Sort numbers
                numbers_in_cluster.sort(key=lambda x: int(x[1]))
                
                # Group numbers into ranges
                number_ranges = []
                current_range = []
                current_start = None
                
                for point, token, token_id in numbers_in_cluster:
                    num = int(token)
                    if current_start is None:
                        current_start = num
                        current_range = [(point, token, token_id)]
                    elif num == current_start + len(current_range):
                        current_range.append((point, token, token_id))
                    else:
                        # End current range
                        if current_range:
                            number_ranges.append((current_start, current_range))
                        # Start new range
                        current_start = num
                        current_range = [(point, token, token_id)]
                
                # Add last range
                if current_range:
                    number_ranges.append((current_start, current_range))
                
                # Label number ranges
                for start_num, range_points in number_ranges:
                    if len(range_points) == 1:
                        # Single number - show it
                        point, token, token_id = range_points[0]
                        ax.annotate(f"{token}", 
                                   (point[0], point[1]), 
                                   xytext=(8, 8), textcoords='offset points',
                                   fontsize=9, alpha=0.9, fontweight='bold', color='green',
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', alpha=0.7, edgecolor='green'))
                    else:
                        # Range - show range and position
                        end_num = start_num + len(range_points) - 1
                        # Use the middle point of the range
                        mid_idx = len(range_points) // 2
                        mid_point, mid_token, mid_token_id = range_points[mid_idx]
                        
                        ax.annotate(f"{start_num}-{end_num}", 
                                   (mid_point[0], mid_point[1]), 
                                   xytext=(10, 10), textcoords='offset points',
                                   fontsize=10, alpha=0.9, fontweight='bold', color='green',
                                   bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.8, edgecolor='green', linewidth=1.5))
                        
                        # Show count of numbers in this range
                        ax.annotate(f"({len(range_points)} nums)", 
                                   (mid_point[0], mid_point[1]), 
                                   xytext=(10, -5), textcoords='offset points',
                                   fontsize=8, alpha=0.7, color='darkgreen')
        
        ax.set_title(f"{title} - Mathematical Tokens", fontsize=16, fontweight='bold')
        ax.set_xlabel("Component 1", fontsize=12)
        ax.set_ylabel("Component 2", fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add legend for token types
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Special Tokens'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='Operations'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=10, label='Numbers')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Mathematical token visualization saved to {output_path}")
        
        return cluster_labels
    
    def plot_focused_clusters(self, embeddings_2d: np.ndarray, token_info: Dict, 
                             cluster_labels: np.ndarray, output_path: str, 
                             title: str = "Focused Cluster Analysis"):
        """Create focused visualization of specific clusters with detailed labeling."""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        # Get unique clusters
        unique_clusters = np.unique(cluster_labels)
        n_clusters = len(unique_clusters)
        
        # Color scheme for clusters
        cluster_colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
        
        for i, cluster_id in enumerate(unique_clusters):
            if i >= 4:  # Only show first 4 clusters
                break
                
            ax = axes[i]
            cluster_mask = cluster_labels == cluster_id
            cluster_points = embeddings_2d[cluster_mask]
            cluster_tokens = [token_info['token_text'][j] for j, m in enumerate(cluster_mask) if m]
            cluster_ids = [token_info['token_id'][j] for j, m in enumerate(cluster_mask) if m]
            cluster_categories = [token_info['category'][j] for j, m in enumerate(cluster_mask) if m]
            
            # Plot cluster points
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                      c=[cluster_colors[i]], alpha=0.8, s=50)
            
            # Add labels for all tokens in this cluster
            for j, (point, token, token_id, category) in enumerate(zip(cluster_points, cluster_tokens, cluster_ids, cluster_categories)):
                # Clean up token text for display
                display_text = token.replace('\n', '\\n').replace('\t', '\\t')
                if len(display_text) > 20:
                    display_text = display_text[:17] + '...'
                
                # Color code by category
                if category == 'alphabetic':
                    color = 'blue'
                elif category == 'numeric':
                    color = 'red'
                elif category == 'punctuation':
                    color = 'green'
                elif category == 'whitespace':
                    color = 'gray'
                else:
                    color = 'black'
                
                ax.annotate(f"{token_id}:{display_text}", 
                           (point[0], point[1]), 
                           xytext=(3, 3), textcoords='offset points',
                           fontsize=8, color=color, alpha=0.9,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
            
            # Add cluster statistics
            category_counts = {}
            for cat in cluster_categories:
                category_counts[cat] = category_counts.get(cat, 0) + 1
            
            stats_text = f"Cluster {cluster_id}\nSize: {len(cluster_points)}\n"
            for cat, count in category_counts.items():
                stats_text += f"{cat}: {count}\n"
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
            
            ax.set_title(f"Cluster {cluster_id} - {len(cluster_points)} tokens")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_clusters, 4):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Focused cluster analysis saved to {output_path}")
    
    def plot_embeddings_3d(self, embeddings_3d: np.ndarray, token_info: Dict, 
                          output_path: str, title: str = "Token Embeddings 3D"):
        """Create 3D scatter plot of embeddings."""
        fig = go.Figure()
        
        # Color by category
        categories = token_info['category']
        unique_categories = list(set(categories))
        colors = px.colors.qualitative.Set3[:len(unique_categories)]
        
        for i, category in enumerate(unique_categories):
            mask = [c == category for c in categories]
            fig.add_trace(go.Scatter3d(
                x=embeddings_3d[mask, 0],
                y=embeddings_3d[mask, 1],
                z=embeddings_3d[mask, 2],
                mode='markers',
                name=category,
                marker=dict(size=3, color=colors[i], opacity=0.7),
                text=[f"ID: {tid}<br>Token: {repr(tt)}" 
                      for tid, tt in zip(np.array(token_info['token_id'])[mask], 
                                        np.array(token_info['token_text'])[mask])],
                hovertemplate='%{text}<extra></extra>'
            ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="Component 1",
                yaxis_title="Component 2",
                zaxis_title="Component 3"
            ),
            width=1000,
            height=800
        )
        
        fig.write_html(output_path)
        print(f"3D plot saved to {output_path}")
    
    def plot_distance_heatmap(self, distances: np.ndarray, token_info: Dict, 
                             output_path: str, max_tokens: int = 100):
        """Create heatmap of pairwise distances."""
        if len(distances) > max_tokens:
            # Sample tokens for heatmap
            indices = np.random.choice(len(distances), max_tokens, replace=False)
            distances_subset = distances[np.ix_(indices, indices)]
            token_ids_subset = [token_info['token_id'][i] for i in indices]
            token_texts_subset = [token_info['token_text'][i] for i in indices]
        else:
            distances_subset = distances
            token_ids_subset = token_info['token_id']
            token_texts_subset = token_info['token_text']
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(distances_subset, 
                   xticklabels=[f"{tid}<br>{repr(tt)[:10]}" for tid, tt in zip(token_ids_subset, token_texts_subset)],
                   yticklabels=[f"{tid}<br>{repr(tt)[:10]}" for tid, tt in zip(token_ids_subset, token_texts_subset)],
                   cmap='viridis', square=True)
        plt.title(f"Pairwise Distances ({self.embedding_space.capitalize()} Space)")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Distance heatmap saved to {output_path}")
    
    def plot_embedding_statistics(self, embeddings: np.ndarray, token_info: Dict, 
                                output_path: str):
        """Plot embedding statistics."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Norm distribution
        norms = np.linalg.norm(embeddings, axis=1)
        axes[0, 0].hist(norms, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title("Embedding Norm Distribution")
        axes[0, 0].set_xlabel("Norm")
        axes[0, 0].set_ylabel("Frequency")
        
        # 2. Norm by category
        categories = token_info['category']
        unique_categories = list(set(categories))
        category_norms = [norms[[c == cat for c in categories]] for cat in unique_categories]
        axes[0, 1].boxplot(category_norms, labels=unique_categories)
        axes[0, 1].set_title("Norm Distribution by Category")
        axes[0, 1].set_ylabel("Norm")
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Distance to origin
        if self.embedding_space in ['poincare', 'lorentz']:
            origin_distances = norms  # For hyperbolic spaces, norm is distance to origin
        else:
            origin_distances = norms
        axes[0, 2].hist(origin_distances, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 2].set_title("Distance to Origin")
        axes[0, 2].set_xlabel("Distance")
        axes[0, 2].set_ylabel("Frequency")
        
        # 4. Embedding dimension variance
        dim_vars = np.var(embeddings, axis=0)
        axes[1, 0].plot(dim_vars)
        axes[1, 0].set_title("Variance by Dimension")
        axes[1, 0].set_xlabel("Dimension")
        axes[1, 0].set_ylabel("Variance")
        
        # 5. Category distribution
        category_counts = [categories.count(cat) for cat in unique_categories]
        axes[1, 1].pie(category_counts, labels=unique_categories, autopct='%1.1f%%')
        axes[1, 1].set_title("Token Category Distribution")
        
        # 6. Distance distribution
        # Sample some distances for efficiency
        if len(embeddings) > 1000:
            sample_indices = np.random.choice(len(embeddings), 1000, replace=False)
            sample_embeddings = embeddings[sample_indices]
        else:
            sample_embeddings = embeddings
        
        distances = self.compute_distances(sample_embeddings)
        # Get upper triangle (excluding diagonal)
        upper_tri = distances[np.triu_indices_from(distances, k=1)]
        axes[1, 2].hist(upper_tri, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 2].set_title("Pairwise Distance Distribution")
        axes[1, 2].set_xlabel("Distance")
        axes[1, 2].set_ylabel("Frequency")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Statistics plot saved to {output_path}")
    
    def create_interactive_plot(self, embeddings_2d: np.ndarray, token_info: Dict, 
                               output_path: str, title: str = "Interactive Token Embeddings"):
        """Create an interactive plot with hover information."""
        df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'token_id': token_info['token_id'],
            'token_text': token_info['token_text'],
            'token_repr': token_info['token_repr'],
            'category': token_info['category']
        })
        
        fig = px.scatter(df, x='x', y='y', color='category',
                        hover_data=['token_id', 'token_repr'],
                        title=title,
                        color_discrete_sequence=px.colors.qualitative.Set3)
        
        fig.update_traces(marker=dict(size=5, opacity=0.7))
        fig.update_layout(
            width=1000,
            height=800,
            showlegend=True
        )
        
        fig.write_html(output_path)
        print(f"Interactive plot saved to {output_path}")
    
    def analyze_token_clusters(self, embeddings: np.ndarray, token_info: Dict, 
                              n_clusters: int = 10) -> Dict:
        """Analyze token clusters using K-means."""
        print(f"Performing K-means clustering with {n_clusters} clusters...")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Analyze clusters
        cluster_analysis = {}
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_tokens = [token_info['token_text'][i] for i, mask in enumerate(cluster_mask) if mask]
            cluster_categories = [token_info['category'][i] for i, mask in enumerate(cluster_mask) if mask]
            
            cluster_analysis[cluster_id] = {
                'size': len(cluster_tokens),
                'tokens': cluster_tokens[:20],  # First 20 tokens
                'categories': cluster_categories,
                'category_counts': {cat: cluster_categories.count(cat) for cat in set(cluster_categories)}
            }
        
        return cluster_analysis
    
    def save_cluster_analysis(self, cluster_analysis: Dict, output_path: str):
        """Save cluster analysis to a text file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Token Embedding Cluster Analysis\n")
            f.write("=" * 50 + "\n\n")
            
            for cluster_id, analysis in cluster_analysis.items():
                f.write(f"Cluster {cluster_id} (Size: {analysis['size']})\n")
                f.write("-" * 30 + "\n")
                f.write(f"Sample tokens: {analysis['tokens']}\n")
                f.write(f"Category distribution: {analysis['category_counts']}\n")
                f.write("\n")
        
        print(f"Cluster analysis saved to {output_path}")
    
    def visualize_all(self, output_dir: str, n_samples: int = 2000, 
                     reduction_method: str = 'tsne', random_seed: int = 42):
        """Generate all visualizations."""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating visualizations for {self.embedding_space} embeddings...")
        
        # Sample tokens
        embeddings, token_info = self.sample_tokens(n_samples, random_seed)
        
        # 1. 2D visualization
        print("Creating 2D visualization...")
        embeddings_2d = self.reduce_dimensions(embeddings, method=reduction_method, 
                                             n_components=2, random_state=random_seed)
        self.plot_embeddings_2d(
            embeddings_2d, token_info,
            os.path.join(output_dir, f"embeddings_2d_{reduction_method}.png"),
            f"Token Embeddings ({self.embedding_space.capitalize()} Space) - {reduction_method.upper()}"
        )
        
        # 1.5. 2D visualization with labels and clustering
        print("Creating 2D visualization with labels and clustering...")
        cluster_labels = self.plot_embeddings_2d_with_labels(
            embeddings_2d, token_info,
            os.path.join(output_dir, f"embeddings_2d_with_labels_{reduction_method}.png"),
            f"Token Embeddings with Labels ({self.embedding_space.capitalize()} Space) - {reduction_method.upper()}"
        )
        
        # 2. 3D visualization
        print("Creating 3D visualization...")
        embeddings_3d = self.reduce_dimensions(embeddings, method=reduction_method, 
                                             n_components=3, random_state=random_seed)
        self.plot_embeddings_3d(
            embeddings_3d, token_info,
            os.path.join(output_dir, f"embeddings_3d_{reduction_method}.html"),
            f"Token Embeddings 3D ({self.embedding_space.capitalize()} Space) - {reduction_method.upper()}"
        )
        
        # 3. Interactive plot
        print("Creating interactive plot...")
        self.create_interactive_plot(
            embeddings_2d, token_info,
            os.path.join(output_dir, f"embeddings_interactive_{reduction_method}.html"),
            f"Interactive Token Embeddings ({self.embedding_space.capitalize()} Space)"
        )
        
        # 4. Distance heatmap
        print("Creating distance heatmap...")
        distances = self.compute_distances(embeddings)
        self.plot_distance_heatmap(
            distances, token_info,
            os.path.join(output_dir, "distance_heatmap.png")
        )
        
        # 5. Statistics
        print("Creating statistics plots...")
        self.plot_embedding_statistics(
            embeddings, token_info,
            os.path.join(output_dir, "embedding_statistics.png")
        )
        
        # 6. Cluster analysis
        print("Performing cluster analysis...")
        cluster_analysis = self.analyze_token_clusters(embeddings, token_info)
        self.save_cluster_analysis(
            cluster_analysis,
            os.path.join(output_dir, "cluster_analysis.txt")
        )
        
        # 7. Focused cluster analysis
        print("Creating focused cluster analysis...")
        self.plot_focused_clusters(
            embeddings_2d, token_info,
            cluster_labels,
            os.path.join(output_dir, "focused_cluster_analysis.png"),
            f"Focused Cluster Analysis ({self.embedding_space.capitalize()} Space)"
        )
        
        # 8. Clean clustering with token names
        print("Creating clean clustering with token names...")
        self.plot_clustering_with_token_names(
            embeddings_2d, token_info,
            os.path.join(output_dir, "clustering_with_token_names.png"),
            f"Token Clustering with Names ({self.embedding_space.capitalize()} Space)"
        )
        
        # 9. Mathematical token visualization
        print("Creating mathematical token visualization...")
        self.plot_mathematical_tokens(
            embeddings_2d, token_info,
            os.path.join(output_dir, "mathematical_token_visualization.png"),
            f"Mathematical Token Embeddings ({self.embedding_space.capitalize()} Space)"
        )
        
        # 10. Zero embedding visualization
        print("Creating zero embedding visualization...")
        zero_analysis = self.plot_with_zero_embedding(
            embeddings_2d, token_info,
            os.path.join(output_dir, "zero_embedding_visualization.png"),
            f"Zero Embedding Analysis ({self.embedding_space.capitalize()} Space)"
        )
        
        print(f"All visualizations saved to {output_dir}")
        
        return {
            'embeddings': embeddings,
            'token_info': token_info,
            'cluster_analysis': cluster_analysis,
            'embedding_space': self.embedding_space,
            'cluster_labels': cluster_labels,
            'zero_analysis': zero_analysis
        }
    
    def plot_with_zero_embedding(self, embeddings_2d: np.ndarray, token_info: Dict, 
                                output_path: str, title: str = "Embeddings with Zero Position"):
        """Visualize embeddings and highlight the zero embedding position."""
        from sklearn.cluster import KMeans
        from scipy.spatial import ConvexHull
        
        # Add zero embedding to the data
        zero_embedding_2d = np.array([[0.0, 0.0]])  # Zero embedding at origin
        embeddings_with_zero = np.vstack([embeddings_2d, zero_embedding_2d])
        
        # Add zero token info
        zero_token_info = {
            'token_id': token_info['token_id'] + [-1],  # Use -1 as special ID for zero
            'token_text': token_info['token_text'] + ['[ZERO]'],
            'token_repr': token_info['token_repr'] + ["'[ZERO]'"],
            'category': token_info['category'] + ['zero']
        }
        
        # Perform clustering on data including zero embedding
        n_clusters = min(8, len(embeddings_with_zero) // 10)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_with_zero)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot 1: Show all embeddings with zero position highlighted
        # Plot regular embeddings
        scatter = ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                             c=cluster_labels[:-1], cmap='tab10', alpha=0.6, s=30)
        
        # Highlight the zero embedding
        zero_idx = len(embeddings_2d)  # Index of zero embedding
        zero_point = embeddings_with_zero[zero_idx]
        zero_cluster = cluster_labels[zero_idx]
        
        ax1.scatter(zero_point[0], zero_point[1], c='red', s=300, marker='*', 
                   edgecolors='black', linewidth=3, label='Zero Embedding', zorder=10)
        
        # Add label for zero embedding
        ax1.annotate(f"ZERO EMBEDDING\nCluster {zero_cluster}", 
                    (zero_point[0], zero_point[1]), 
                    xytext=(20, 20), textcoords='offset points',
                    fontsize=14, fontweight='bold', color='red',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.9, edgecolor='red', linewidth=2),
                    ha='center')
        
        # Add origin marker
        ax1.scatter(0, 0, c='black', s=100, marker='+', linewidths=3, label='Origin (0,0)', zorder=5)
        
        # Draw circle around zero embedding to show proximity to origin
        circle = plt.Circle((0, 0), 0.1, fill=False, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax1.add_patch(circle)
        
        ax1.set_title(f"{title} - Zero Embedding Added")
        ax1.set_xlabel("Component 1")
        ax1.set_ylabel("Component 2")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Show distance to zero embedding
        # Calculate distances from all points to the zero embedding
        distances_to_zero = np.linalg.norm(embeddings_with_zero - zero_point, axis=1)
        
        # Color points by distance to zero (excluding zero itself for better color scale)
        distances_to_zero_regular = distances_to_zero[:-1]
        scatter2 = ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                              c=distances_to_zero_regular, cmap='viridis', alpha=0.7, s=30)
        
        # Highlight zero embedding again
        ax2.scatter(zero_point[0], zero_point[1], c='red', s=300, marker='*', 
                   edgecolors='black', linewidth=3, label='Zero Embedding', zorder=10)
        
        # Add colorbar for distance
        cbar = plt.colorbar(scatter2, ax=ax2)
        cbar.set_label('Distance to Zero Embedding')
        
        # Find closest and farthest tokens to zero (excluding zero itself)
        distances_to_zero_regular = distances_to_zero[:-1]
        closest_idx = np.argmin(distances_to_zero_regular)
        farthest_idx = np.argmax(distances_to_zero_regular)
        
        closest_point = embeddings_2d[closest_idx]
        farthest_point = embeddings_2d[farthest_idx]
        
        # Label closest token
        closest_token = token_info['token_text'][closest_idx]
        closest_token_id = token_info['token_id'][closest_idx]
        ax2.annotate(f"Closest: {closest_token} (ID:{closest_token_id})", 
                    (closest_point[0], closest_point[1]), 
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=10, fontweight='bold', color='green',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8, edgecolor='green'))
        
        # Label farthest token
        farthest_token = token_info['token_text'][farthest_idx]
        farthest_token_id = token_info['token_id'][farthest_idx]
        ax2.annotate(f"Farthest: {farthest_token} (ID:{farthest_token_id})", 
                    (farthest_point[0], farthest_point[1]), 
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=10, fontweight='bold', color='purple',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.8, edgecolor='purple'))
        
        ax2.set_title(f"{title} - Distance to Zero Embedding")
        ax2.set_xlabel("Component 1")
        ax2.set_ylabel("Component 2")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Zero embedding visualization saved to {output_path}")
        
        # Print statistics
        print(f"\nZero Embedding Analysis:")
        print(f"- Zero embedding added at position (0, 0)")
        print(f"- Zero embedding assigned to cluster {zero_cluster}")
        print(f"- Closest token to zero: '{closest_token}' (ID: {closest_token_id})")
        print(f"- Distance to closest: {distances_to_zero_regular[closest_idx]:.4f}")
        print(f"- Farthest token from zero: '{farthest_token}' (ID: {farthest_token_id})")
        print(f"- Distance to farthest: {distances_to_zero_regular[farthest_idx]:.4f}")
        print(f"- Average distance to zero: {np.mean(distances_to_zero_regular):.4f}")
        print(f"- Standard deviation of distances: {np.std(distances_to_zero_regular):.4f}")
        
        return {
            'zero_idx': zero_idx,
            'zero_token': '[ZERO]',
            'zero_token_id': -1,
            'distances_to_zero': distances_to_zero_regular,
            'closest_idx': closest_idx,
            'farthest_idx': farthest_idx,
            'zero_cluster': zero_cluster,
            'avg_distance_to_zero': np.mean(distances_to_zero_regular),
            'std_distance_to_zero': np.std(distances_to_zero_regular)
        }


def main():
    parser = argparse.ArgumentParser(description="Visualize token embeddings from language models")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the model directory")
    parser.add_argument("--output_dir", type=str, default="embedding_visualizations",
                       help="Output directory for visualizations")
    parser.add_argument("--n_samples", type=int, default=2000,
                       help="Number of tokens to sample for visualization")
    parser.add_argument("--reduction_method", type=str, default="tsne", choices=["tsne", "pca"],
                       help="Dimensionality reduction method")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                       help="Device to load the model on")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--max_labels", type=int, default=100,
                       help="Maximum number of token labels to show in plots")
    parser.add_argument("--focused_only", action="store_true",
                       help="Only generate focused cluster analysis (faster)")
    parser.add_argument("--raw_embeddings", action="store_true",
                       help="Use raw embeddings instead of sampled tokens")
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = EmbeddingVisualizer(args.model_path, args.device, args.raw_embeddings)
    
    if args.focused_only:
        # Quick focused analysis
        print("Performing focused cluster analysis only...")
        embeddings, token_info = visualizer.sample_tokens(args.n_samples, args.random_seed)
        embeddings_2d = visualizer.reduce_dimensions(embeddings, method=args.reduction_method, 
                                                   n_components=2, random_state=args.random_seed)
        cluster_labels = visualizer.plot_embeddings_2d_with_labels(
            embeddings_2d, token_info,
            os.path.join(args.output_dir, f"embeddings_2d_with_labels_{args.reduction_method}.png"),
            f"Token Embeddings with Labels ({visualizer.embedding_space.capitalize()} Space) - {args.reduction_method.upper()}",
            max_labels=args.max_labels
        )
        visualizer.plot_focused_clusters(
            embeddings_2d, token_info,
            cluster_labels,
            os.path.join(args.output_dir, "focused_cluster_analysis.png"),
            f"Focused Cluster Analysis ({visualizer.embedding_space.capitalize()} Space)"
        )
        visualizer.plot_clustering_with_token_names(
            embeddings_2d, token_info,
            os.path.join(args.output_dir, "clustering_with_token_names.png"),
            f"Token Clustering with Names ({visualizer.embedding_space.capitalize()} Space)"
        )
        visualizer.plot_mathematical_tokens(
            embeddings_2d, token_info,
            os.path.join(args.output_dir, "mathematical_token_visualization.png"),
            f"Mathematical Token Embeddings ({visualizer.embedding_space.capitalize()} Space)"
        )
        zero_analysis = visualizer.plot_with_zero_embedding(
            embeddings_2d, token_info,
            os.path.join(args.output_dir, "zero_embedding_visualization.png"),
            f"Zero Embedding Analysis ({visualizer.embedding_space.capitalize()} Space)"
        )
        print(f"Focused analysis saved to {args.output_dir}")
        
        # Print zero embedding summary
        print(f"\nZero Embedding Summary:")
        print(f"- Zero token: '{zero_analysis['zero_token']}' (ID: {zero_analysis['zero_token_id']})")
        print(f"- Farthest token: '{token_info['token_text'][zero_analysis['farthest_idx']]}' (ID: {token_info['token_id'][zero_analysis['farthest_idx']]})")
    else:
        # Generate all visualizations
        results = visualizer.visualize_all(
            args.output_dir,
            n_samples=args.n_samples,
            reduction_method=args.reduction_method,
            random_seed=args.random_seed
        )
        
        print("\nVisualization complete!")
        print(f"Results saved to: {args.output_dir}")
        print(f"Embedding space: {results['embedding_space']}")
        print(f"Number of tokens visualized: {len(results['embeddings'])}")
        print(f"Number of clusters found: {len(np.unique(results['cluster_labels']))}")


if __name__ == "__main__":
    main() 