from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import numpy as np

def resize_images(samples, target_size=(75, 75)):
    """Resize 32x32 images to 75x75."""
    resized_samples = np.zeros((samples.shape[0], 3, *target_size), dtype=np.float32)
    for i in range(samples.shape[0]):
        img = array_to_img(samples[i].transpose(1, 2, 0))  # Convert to HWC for resizing
        img = img.resize(target_size)
        resized_samples[i] = img_to_array(img).transpose(2, 0, 1)  # Convert back to CHW
    return resized_samples

def compute_features(samples):
    """Extract features from InceptionV3 for the given samples."""
    # Resize images to 75x75 and preprocess for InceptionV3
    samples_resized = resize_images(samples, target_size=(75, 75))
    samples_resized = samples_resized.transpose(0, 2, 3, 1)  # Convert to NHWC
    samples_resized = preprocess_input(samples_resized * 255)  # Scale to [0, 255]
    
    # Load InceptionV3 and extract features
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(75, 75, 3))
    features = model.predict(samples_resized, batch_size=64, verbose=1)
    return features

def compute_diversity_metrics(features):
    """Calculate diversity metrics: feature variance and pairwise distances."""
    variance = np.var(features, axis=0).mean()
    pairwise_dist = pairwise_distances(features)
    mean_pairwise_dist = pairwise_dist[np.triu_indices(len(features), k=1)].mean()
    return variance, mean_pairwise_dist

def analyze_mode_collapse(features, num_clusters=10):
    """Perform clustering to analyze mode collapse."""
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(features)
    cluster_counts = np.bincount(kmeans.labels_)
    return cluster_counts

# Load your 10k samples
samples = np.load("experiments/jan_experiments/10000_samples_epoch_1000.npy")  # Replace with actual file path

# Step 1: Compute Features
features = compute_features(samples)

# Step 2: Compute Diversity Metrics
feature_variance, mean_pairwise_distance = compute_diversity_metrics(features)
print(f"Feature Variance: {feature_variance}")
print(f"Mean Pairwise Feature Distance: {mean_pairwise_distance}")

# Step 3: Analyze Mode Collapse
cluster_distribution = analyze_mode_collapse(features)
print(f"Cluster Distribution: {cluster_distribution}")
