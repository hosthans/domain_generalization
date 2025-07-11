import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from tqdm.notebook import tqdm
import os
from sklearn.decomposition import PCA

# PACS class names
PACS_CLASSES = ['dog', 'elephant', 'giraffe',
                'guitar', 'horse', 'house', 'person']


def extract_features(model, dataloader, max_samples=None):
    """
    Extract features from the layer of a ResNet model.

    Args:
        model: Trained PyTorch model
        dataloader: Test data loader
        max_samples: Maximum samples to process (None = use all samples)

    Returns:
        features, labels
    """
    use_all_samples = max_samples is None

    model.eval()
    features = []
    labels = []

    sample_count = 0

    device = next(model.parameters()).device

    with torch.no_grad():
        for batch_idx, (data, _domain, target) in enumerate(tqdm(dataloader, desc="Extracting features")):
            if not use_all_samples and sample_count >= max_samples:
                break

            # Limit batch size
            if use_all_samples:
                actual_batch_size = data.size(0)
            else:
                actual_batch_size = min(
                    data.size(0), max_samples - sample_count)

            data = data[:actual_batch_size].to(device)
            target = target[:actual_batch_size].to(device)

            try:
                # For ResNet models, get features before the final classification layer
                if hasattr(model, 'featuremaps'):
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

                feat = feat.cpu().numpy()
                if len(feat.shape) > 2:
                    feat = feat.reshape(feat.shape[0], -1)

                features.append(feat)
                labels.extend(target.cpu().numpy())
                sample_count += actual_batch_size

            except Exception as e:
                print(f"Error in feature extraction: {e}")
                break

    if len(features) == 0:
        print("No features extracted!")
        return np.array([]), np.array([])

    features = np.vstack(features)
    labels = np.array(labels)

    # Check array length
    min_length = min(len(features), len(labels))
    features = features[:min_length]
    labels = labels[:min_length]

    return features, labels


def visualize_models(models, test_loader, target_class=0, save_path=None, max_samples=1000):
    """

    Args:
        models: Either a single model OR {'name': model} dictionary
        test_loader: Test data loader
        target_class: Class left out (0-6)
        save_path: Save path
        max_samples: Max samples for t-SNE (1000 default for efficiency)

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
    class_names = [cls for i, cls in enumerate(
        PACS_CLASSES) if i != target_class]

    for idx, (name, model) in enumerate(models.items()):

        features, labels = extract_features(
            model, test_loader, max_samples=max_samples)

        # adjust t-SNE parameters based on dataset size
        perplexity = min(30, len(features)-1, 50)
        n_iter = 1000 if len(features) > 500 else 500

        embedding = TSNE(n_components=2, perplexity=perplexity,
                         max_iter=n_iter, random_state=42).fit_transform(features)

        ax = axes[idx]
        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            class_name = class_names[label] if label < len(
                class_names) else f"Class {label}"
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=[colors[i]], label=class_name, alpha=0.7, s=40)

        # silhouette score, the score is normalised between -1 and 1 and measueres how similar an obeject is to its own cluster compared to other clusters
        try:
            score = silhouette_score(embedding, labels)
            results[name] = score
        except:
            score = 0.0
            results[name] = score

        ax.set_title(
            f"{name}\nSilhouette: {score:.3f} ({len(features)} samples)", fontsize=12)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        if idx < len(axes):
            axes[idx].set_visible(False)

    # Legend
    if n_models > 0:
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    plt.suptitle(
        f"Model Analysis - Left out: {PACS_CLASSES[target_class]}", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    if len(results) > 1:
        print("\nResults:")
        for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"{name}: {score:.4f}")

    return fig, results


def extract_layer_features(model, dataloader, layer_name, max_samples=1000):
    """
    Extract features from a specific layer during training.

    Args:
        model: Your model
        dataloader: Data loader
        layer_name: e.g., 'layer1', 'layer2', 'layer3', 'layer4'
        max_samples: Number of samples (1000 default for efficiency)

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

    target_layer = getattr(model, layer_name)
    handle = target_layer.register_forward_hook(get_activation(layer_name))

    device = next(model.parameters()).device
    sample_count = 0
    use_all_samples = max_samples is None

    with torch.no_grad():
        for data, _domain, target in dataloader:
            if not use_all_samples and sample_count >= max_samples:
                break

            if use_all_samples:
                batch_size = data.size(0)
            else:
                batch_size = min(data.size(0), max_samples - sample_count)

            data = data[:batch_size].to(device)
            target = target[:batch_size]

            # Forward pass (triggers hook)
            _ = model(data)

            # Get layer activation
            layer_output = activation[layer_name]

            # Global average pooling if needed
            if len(layer_output.shape) == 4:
                layer_output = torch.mean(layer_output, dim=[2, 3])

            features.append(layer_output.cpu().numpy())
            labels.extend(target.numpy())
            sample_count += data.size(0)

    handle.remove()

    features = np.vstack(features)
    labels = np.array(labels)

    return features, labels


def visualize_layer(model, test_loader, layer_name, target_class=0):
    """Visualize specific layer during training (uses 1000 samples for efficiency)"""
    features, labels = extract_layer_features(
        model, test_loader, layer_name, max_samples=1000)

    # Adjust t-SNE parameters based on dataset size
    perplexity = min(30, len(features)-1, 50)
    n_iter = 1000 if len(features) > 500 else 500

    embedding = TSNE(n_components=2, perplexity=perplexity,
                     max_iter=n_iter, random_state=42).fit_transform(features)

    plt.figure(figsize=(8, 6))
    class_names = [cls for i, cls in enumerate(
        PACS_CLASSES) if i != target_class]
    unique_labels = np.unique(labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        class_name = class_names[label] if label < len(
            class_names) else f"Class {label}"
        plt.scatter(embedding[mask, 0], embedding[mask, 1],
                    c=[colors[i]], label=class_name, alpha=0.7, s=40)

    try:
        score = silhouette_score(embedding, labels)
        plt.title(
            f"{layer_name} - Silhouette: {score:.3f} ({len(features)} samples)")
    except:
        plt.title(f"{layer_name} ({len(features)} samples)")

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def track_feature_evolution(model, test_loader, epoch, target_domain,
                            results_tracker=None, visualize_every=5, save_dir=None, max_samples=1000):
    """

    Args:
        model: Current model
        test_loader: Test data loader
        epoch: Current epoch
        target_domain: Target domain name
        results_tracker: Dict to store silhouette scores over time
        visualize_every: Show plot every N epochs
        save_dir: Directory to save plots (optional)
        max_samples: Max samples for t-SNE (1000 default for efficiency), as t-SNE takes especially a lot of RAM and time

    Returns:
        results_tracker: Updated tracker dictionary
    """
    if results_tracker is None:
        results_tracker = {
            'epochs': [], 'silhouette_scores': [], 'target_domain': target_domain}

    features, labels = extract_features(
        model, test_loader, max_samples=max_samples)

    if len(features) == 0:
        return results_tracker

    # Adjust t-SNE settings based on dataset size
    perplexity = min(30, len(features)-1, 50)
    n_iter = 1000 if len(features) > 500 else 500

    embedding = TSNE(n_components=2, perplexity=perplexity,
                     max_iter=n_iter, random_state=42).fit_transform(features)

    # Calculate silhouette score
    try:
        sil_score = silhouette_score(embedding, labels)
    except Exception as e:
        sil_score = 0.0

    results_tracker['epochs'].append(epoch)
    results_tracker['silhouette_scores'].append(sil_score)

    # Visualize only periodically or at the last epoch
    should_visualize = (epoch % visualize_every == 0) or (epoch == 1)

    if should_visualize:
        # Create plot - only tsne
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.set_visible(False)

        class_names = [cls for i, cls in enumerate(
            PACS_CLASSES) if i != target_domain] if isinstance(target_domain, int) else PACS_CLASSES
        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))

        # Plot t-SNE (we already have the embedding)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            class_name = class_names[label] if label < len(
                class_names) else f"Class {label}"
            ax2.scatter(embedding[mask, 0], embedding[mask, 1],
                        c=[colors[i]], label=class_name, alpha=0.7, s=40)

        ax2.set_title(f't-SNE - Epoch {epoch}\nSilhouette: {sil_score:.3f}')
        ax2.set_xlabel('t-SNE Dim 1 (relative positions)')
        ax2.set_ylabel('t-SNE Dim 2 (relative positions)')
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

        plt.suptitle(
            f"Feature Analysis - Target: {target_domain} ({len(features)} samples)", fontsize=14)
        plt.tight_layout()

        # Save if requested
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(
                save_dir, f"epoch_{epoch:02d}_{target_domain}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

    return results_tracker


def plot_feature_evolution(results_tracker):
    """
    Plot the evolution of feature separability over training epochs.

    Args:
        results_tracker: Dictionary from track_feature_evolution()
    """
    plt.figure(figsize=(10, 6))
    plt.plot(results_tracker['epochs'], results_tracker['silhouette_scores'],
             'b-o', linewidth=2, markersize=6)
    plt.xlabel('Epoch')
    plt.ylabel('Silhouette Score')
    plt.title(
        f'Feature Separability Evolution - Target: {results_tracker["target_domain"]}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
