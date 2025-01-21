import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


def calculate_histograms(data, eps_values, min_samples_values):

    nearest_neighbor_distances = []
    for i in range(len(data)):
        distances = np.linalg.norm(data - data[i], axis=1)
        distances = np.sort(distances)[1]
        nearest_neighbor_distances.append(distances)
    plt.hist(nearest_neighbor_distances, bins=20, edgecolor='k')
    plt.title("Histogram of Distances to the Nearest Neighbor")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.show()

    for eps in eps_values:
        neighbors_within_eps = []
        for i in range(len(data)):
            distances = np.linalg.norm(data - data[i], axis=1)
            count = np.sum(distances <= eps) - 1
            neighbors_within_eps.append(count)
        plt.hist(neighbors_within_eps, bins=20, edgecolor='k')
        plt.title(f"Histogram of Neighbors within eps={eps}")
        plt.xlabel("Number of Neighbors")
        plt.ylabel("Frequency")
        plt.show()

    for min_samples in min_samples_values:
        minpts_distances = []
        for i in range(len(data)):
            distances = np.linalg.norm(data - data[i], axis=1)
            distances = np.sort(distances)
            if len(distances) > min_samples:
                minpts_distances.append(distances[min_samples])
        plt.hist(minpts_distances, bins=20, edgecolor='k')
        plt.title(f"Histogram of Distance to {min_samples}-th Nearest Neighbor")
        plt.xlabel("Distance")
        plt.ylabel("Frequency")
        plt.show()


def detect_outliers(data, eps, min_samples):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(data)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = 'k'

        class_member_mask = (labels == k)

        xy = data[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)

        xy = data[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=6)

    plt.title('DBSCAN Outlier Detection')
    plt.show()


example_data = np.array([
    [90, -29], [61, -34], [24, 50], [33, 44], [63, -32], [83, -49],
    [11, 34], [87, -34], [87, -39], [2, 9], [53, -18], [24, 34], [21, 1],
    [92, -36], [5, 27], [91, -15], [8, 18], [60, -36], [10, 29], [80, -44],
    [31, 48], [82, -48], [94, -24], [32, 50], [21, 48], [82, -42], [92, -21],
    [26, 48], [68, -44], [77, -50], [56, -27]
])

eps_values = [10]
min_samples_values = [3]
eps = 10
min_samples = 3

calculate_histograms(example_data, eps_values, min_samples_values)

detect_outliers(example_data, eps, min_samples)
