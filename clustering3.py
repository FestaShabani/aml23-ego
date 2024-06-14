import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import Counter

# Define custom dataset class
class MyDataset:
    def __init__(self, file_path):
        with open(file_path, 'rb') as f:
            self.data = pickle.load(f)['features']

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

# Load features and create dataset instance
file_path = 'saved_features/FineTuned_Dense_D1_train_16.pkl'
dataset = MyDataset(file_path)

# Extract features from the dataset and flatten them
features = [sample['features_RGB'].flatten() for sample in dataset]

# Convert features to numpy array
features = np.array(features)

# Perform clustering on the features
kmeans = KMeans(n_clusters=8, random_state=42)
cluster_labels = kmeans.fit_predict(features)

# Find the size of each cluster label
cluster_label_sizes = np.bincount(cluster_labels)

class_label_mapping = {
    0: "take (get)",
    1: "put-down (put/place)",
    2: "open",
    3: "close",
    4: "wash (clean)",
    5: "cut",
    6: "stir (mix)",
    7: "pour"
}

# Load true labels from the dataset
ek_dataframe = pd.read_pickle('train_val/D1_train.pkl')
true_labels = ek_dataframe['verb_class']

# Find the number of occurrences of each class label
class_label_counts = np.bincount(true_labels)

# Print the number of occurrences of each class label
for class_label, count in enumerate(class_label_counts):
    print(f"{class_label_mapping[class_label]}: {count}")

# Initialize the mapping with None for each cluster
cluster_to_class_mapping = [None] * 8

# For each cluster
for cluster in range(8):
    # Find the class labels for samples in this cluster
    class_labels_in_cluster = true_labels[cluster_labels == cluster]
    
    # Count the occurrences of each class label
    class_counts = Counter(class_labels_in_cluster)
    
    # Sort the class labels by their counts in descending order
    sorted_class_labels = [class_label for class_label, count in class_counts.most_common()]
    
    # Find the most common class that hasn't been assigned to a cluster yet
    for class_label in sorted_class_labels:
        if class_label not in cluster_to_class_mapping:
            cluster_to_class_mapping[cluster] = class_label
            break

# Now each cluster should be mapped to a unique class

# Perform t-SNE to reduce dimensionality for visualization
tsne = TSNE(n_components=2, random_state=42)
features_tsne = tsne.fit_transform(features)

# Plot the clusters with their assigned class labels
plt.figure(figsize=(8, 6))
for cluster in range(8):
    plt.scatter(features_tsne[cluster_labels == cluster, 0], 
                features_tsne[cluster_labels == cluster, 1], 
                label=f'Cluster {cluster} ({cluster_to_class_mapping[cluster]})')
plt.title('Clusters Visualization (t-SNE) with Assigned Class Labels')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend()
plt.grid(True)
plt.show()