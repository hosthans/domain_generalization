"""
Simplified notebook utilities - just the essentials
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from tqdm.notebook import tqdm
import os

# PACS class names
PACS_CLASSES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']


def extract_features_simple(model, dataloader, max_samples=1000):
    """
    Extract features from the penultimate layer of a ResNet model.
    
    Args:
        model: Trained PyTorch model
        dataloader: Test data loader
        max_samples: Maximum samples to process
    
    Returns:
        features, labels
    """
    model.eval()
    features = []
    labels = []
    
    sample_count = 0
    
    # Get the device the model is on
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc="Extracting features")):
            if sample_count >= max_samples:
                break
                
            # Limit batch size if we're near the max_samples limit
            actual_batch_size = min(data.size(0), max_samples - sample_count)
            data = data[:actual_batch_size].to(device)
            target = target[:actual_batch_size].to(device)
            
            # For ResNet models, get features before the final classification layer
            if hasattr(model, 'featuremaps'):
                # If model has featuremaps method (from resnet_ms)
                feat = model.featuremaps(data)
            else:
                # For standard ResNet, extract features manually
                x = model.conv1(data)
                x = model.bn1(x)
                x = model.relu(x)
                x = model.maxpool(x)
                x = model.layer1(x)
                x = model.layer2(x)
                x = model.layer3(x)
                x = model.layer4(x)
                feat = model.avgpool(x)
                feat = feat.view(feat.size(0), -1)
            
            # Convert to numpy and flatten
            feat = feat.cpu().numpy()
            if len(feat.shape) > 2:
                feat = feat.reshape(feat.shape[0], -1)
            
            features.append(feat)
            labels.extend(target.cpu().numpy())
            sample_count += actual_batch_size
    
    features = np.vstack(features)
    labels = np.array(labels)
    
    # Ensure all arrays have the same length
    min_length = min(len(features), len(labels))
    features = features[:min_length]
    labels = labels[:min_length]
    
    return features, labels


def visualize_models(models, test_loader, target_class=0, save_path=None):
    """
    Universal visualization function.
    
    Args:
        models: Either a single model OR {'name': model} dictionary
        test_loader: Test data loader
        target_class: Class left out (0-6)
        save_path: Save path
    
    Returns:
        fig, results (silhouette scores)
    """
    # Handle single model case
    if not isinstance(models, dict):
        models = {'Model': models}
    
    n_models = len(models)
    
    # Dynamic subplot layout
    if n_models == 1:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        axes = [ax]
    else:
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_models > 1:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    results = {}
    class_names = [cls for i, cls in enumerate(PACS_CLASSES) if i != target_class]
    
    for idx, (name, model) in enumerate(models.items()):
        print(f"Analyzing {name}...")
        
        # Extract and visualize
        features, labels = extract_features_simple(model, test_loader, max_samples=800)
        embedding = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42).fit_transform(features)
        
        ax = axes[idx]
        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            class_name = class_names[label] if label < len(class_names) else f"Class {label}"
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                      c=[colors[i]], label=class_name, alpha=0.7, s=40)
        
        # Silhouette score
        try:
            score = silhouette_score(embedding, labels)
            results[name] = score
        except:
            score = 0.0
            results[name] = score
        
        ax.set_title(f"{name}\nSilhouette: {score:.3f}", fontsize=12)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        if idx < len(axes):
            axes[idx].set_visible(False)
    
    # Legend on first plot
    if n_models > 0:
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.suptitle(f"Model Analysis - Left out: {PACS_CLASSES[target_class]}", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    
    # Print results
    if len(results) > 1:
        print("\nResults:")
        for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"{name}: {score:.4f}")
    
    return fig, results


def extract_layer_features(model, dataloader, layer_name, max_samples=500):
    """
    Extract features from a specific layer during training.
    
    Args:
        model: Your model
        dataloader: Data loader
        layer_name: e.g., 'layer1', 'layer2', 'layer3', 'layer4'
        max_samples: Number of samples
    
    Returns:
        features, labels
    """
    model.eval()
    features = []
    labels = []
    activation = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # Register hook
    target_layer = getattr(model, layer_name)
    handle = target_layer.register_forward_hook(get_activation(layer_name))
    
    device = next(model.parameters()).device
    sample_count = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            if sample_count >= max_samples:
                break
                
            data = data[:min(data.size(0), max_samples - sample_count)].to(device)
            target = target[:min(target.size(0), max_samples - sample_count)]
            
            # Forward pass (triggers hook)
            _ = model(data)
            
            # Get layer activation
            layer_output = activation[layer_name]
            
            # Global average pooling if needed
            if len(layer_output.shape) == 4:  # Conv layer output
                layer_output = torch.mean(layer_output, dim=[2, 3])
            
            features.append(layer_output.cpu().numpy())
            labels.extend(target.numpy())
            sample_count += data.size(0)
    
    # Clean up
    handle.remove()
    
    features = np.vstack(features)
    labels = np.array(labels)
    
    return features, labels


def visualize_layer(model, test_loader, layer_name, target_class=0):
    """Visualize specific layer during training"""
    print(f"Analyzing {layer_name}...")
    
    features, labels = extract_layer_features(model, test_loader, layer_name, max_samples=500)
    embedding = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42).fit_transform(features)
    
    plt.figure(figsize=(8, 6))
    class_names = [cls for i, cls in enumerate(PACS_CLASSES) if i != target_class]
    unique_labels = np.unique(labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        class_name = class_names[label] if label < len(class_names) else f"Class {label}"
        plt.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[colors[i]], label=class_name, alpha=0.7, s=40)
    
    try:
        score = silhouette_score(embedding, labels)
        plt.title(f"{layer_name} - Silhouette: {score:.3f}")
    except:
        plt.title(f"{layer_name}")
    
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show() 