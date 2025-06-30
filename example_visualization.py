#!/usr/bin/env python3
"""
Example script for token embedding visualization.

This script demonstrates how to use the EmbeddingVisualizer class
to visualize token embeddings from your hyperbolic models.
"""

import os
import sys
from visualize_embeddings import EmbeddingVisualizer

def main():
    # Example usage
    print("Token Embedding Visualization Example")
    print("=" * 50)
    
    # Check if a model path is provided
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Default to a hyperbolic model if available
        model_path = "hyperbolic_models/lorentz_log_exp_all_wo_norm"
        if not os.path.exists(model_path):
            print(f"Model path {model_path} not found.")
            print("Please provide a model path as argument:")
            print("python example_visualization.py <model_path>")
            return
    
    print(f"Using model: {model_path}")
    
    # Create visualizer
    try:
        visualizer = EmbeddingVisualizer(model_path, device='cpu')
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create output directory
    output_dir = "example_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample tokens and create 2D visualization with labels
    print("\nCreating 2D visualization with token labels...")
    embeddings, token_info = visualizer.sample_tokens(n_samples=500, random_seed=42)
    embeddings_2d = visualizer.reduce_dimensions(embeddings, method='tsne', 
                                               n_components=2, random_state=42)
    
    # Create visualization with labels
    cluster_labels = visualizer.plot_embeddings_2d_with_labels(
        embeddings_2d, token_info,
        os.path.join(output_dir, "example_embeddings_with_labels.png"),
        "Example Token Embeddings with Labels",
        max_labels=50  # Show fewer labels for clarity
    )
    
    # Create focused cluster analysis
    print("Creating focused cluster analysis...")
    visualizer.plot_focused_clusters(
        embeddings_2d, token_info,
        cluster_labels,
        os.path.join(output_dir, "example_focused_clusters.png"),
        "Example Focused Cluster Analysis"
    )
    
    # Create interactive plot
    print("Creating interactive plot...")
    visualizer.create_interactive_plot(
        embeddings_2d, token_info,
        os.path.join(output_dir, "example_interactive.html"),
        "Example Interactive Token Embeddings"
    )
    
    print(f"\nVisualizations saved to: {output_dir}")
    print("Files created:")
    print("- example_embeddings_with_labels.png: 2D plot with token labels and clustering")
    print("- example_focused_clusters.png: Detailed view of individual clusters")
    print("- example_interactive.html: Interactive plot (open in browser)")
    
    # Print some statistics
    print(f"\nStatistics:")
    print(f"- Embedding space: {visualizer.embedding_space}")
    print(f"- Number of tokens visualized: {len(embeddings)}")
    print(f"- Number of clusters: {len(set(cluster_labels))}")
    
    # Show some example tokens from each cluster
    print(f"\nExample tokens from each cluster:")
    for cluster_id in sorted(set(cluster_labels)):
        cluster_mask = cluster_labels == cluster_id
        cluster_tokens = [token_info['token_text'][i] for i, m in enumerate(cluster_mask) if m]
        print(f"Cluster {cluster_id}: {cluster_tokens[:5]}...")

if __name__ == "__main__":
    main() 