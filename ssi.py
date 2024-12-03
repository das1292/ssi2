import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skfuzzy.cluster import cmeans


def generate_spiral_data(n_points, noise=0.1):
    t = np.linspace(0, 4 * np.pi, n_points)
    x = t * np.cos(t) + np.random.normal(scale=noise, size=n_points)
    y = t * np.sin(t) + np.random.normal(scale=noise, size=n_points)
    return np.vstack((x, y)).T


def plot_clusters_with_colorful_squares(data, labels, centers, title):
    plt.figure(figsize=(8, 6))

    for i in range(len(centers)):
        cluster_points = data[labels == i]
        if cluster_points.shape[1] > 1:  # 2D
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Grupa {i + 1}',
                        marker='s', edgecolors='black', facecolors=plt.cm.jet(i / len(centers)))
        else:
            plt.scatter(cluster_points[:, 0], np.zeros_like(cluster_points[:, 0]), label=f'Grupa {i + 1}',
                        marker='s', edgecolors='black', facecolors=plt.cm.jet(i / len(centers)))

    if centers.shape[1] > 1:
        plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='o', s=100, label='Środki', zorder=5)
    else:
        plt.scatter(centers[:, 0], np.zeros_like(centers[:, 0]), c='red', marker='o', s=100, label='Środki', zorder=5)

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def kmeans_clustering(data, k=3, max_iter=10, distance_metric="euclidean"):
    if distance_metric == "manhattan":
        centers = data[np.random.choice(data.shape[0], k, replace=False), :]

        for i in range(max_iter):
            dist_matrix = np.abs(data[:, [0]] - centers[:, [0]].T)
            labels = np.argmin(dist_matrix, axis=1)
            new_centers = np.array([data[labels == j].mean(axis=0) for j in range(k)])
            if np.all(new_centers == centers):
                break
            centers = new_centers

        return centers, labels

    else:
        kmeans = KMeans(n_clusters=k, max_iter=max_iter, random_state=123, init='random', algorithm='lloyd')
        kmeans.fit(data)
        return kmeans.cluster_centers_, kmeans.labels_


def fuzzy_c_means(data, c=3, m=2, max_iter=20):
    data_transposed = data.T

    u_initial = np.random.dirichlet(np.ones(c), size=data_transposed.shape[1]).T

    cntr_partial, u_partial, _, _, _, _, _ = cmeans(
        data_transposed, c=c, m=m, error=0.005, maxiter=4, init=u_initial
    )

    cntr_full, u_full, _, _, _, _, _ = cmeans(
        data_transposed, c=c, m=m, error=0.005, maxiter=max_iter, init=u_partial
    )

    u_full_labels = np.argmax(u_full, axis=0)
    high_membership_counts = [(u_full[i] > 0.6).sum() for i in range(c)]

    return cntr_partial, cntr_full, u_full_labels, high_membership_counts


data = generate_spiral_data(n_points=200)

centers_kmeans_euc_4, labels_kmeans_euc_4 = kmeans_clustering(data, k=3, max_iter=4, distance_metric="euclidean")
centers_kmeans_euc_10, labels_kmeans_euc_10 = kmeans_clustering(data, k=3, max_iter=10, distance_metric="euclidean")
plot_clusters_with_colorful_squares(data, labels_kmeans_euc_4, centers_kmeans_euc_4,
                                    "K-średnie (Euklidesowa, 4 iteracje)")
plot_clusters_with_colorful_squares(data, labels_kmeans_euc_10, centers_kmeans_euc_10,
                                    "K-średnie (Euklidesowa, 10 iteracji)")

centers_kmeans_x1x2_4, labels_kmeans_x1x2_4 = kmeans_clustering(data, k=4, max_iter=4, distance_metric="manhattan")
centers_kmeans_x1x2_10, labels_kmeans_x1x2_10 = kmeans_clustering(data, k=4, max_iter=10, distance_metric="manhattan")
plot_clusters_with_colorful_squares(data, labels_kmeans_x1x2_4, centers_kmeans_x1x2_4,
                                    "K-średnie (Manhattan, 4 iteracje)")
plot_clusters_with_colorful_squares(data, labels_kmeans_x1x2_10, centers_kmeans_x1x2_10,
                                    "K-średnie (Manhattan, 10 iteracji)")

centers_fcm_4, centers_fcm_20, labels_fcm_20, high_membership_counts = fuzzy_c_means(data, c=3, m=2, max_iter=20)
plot_clusters_with_colorful_squares(data, labels_fcm_20, centers_fcm_20, "Fuzzy C-Means (20 iteracji)")

centers_fcm_4, centers_fcm_20, labels_fcm_20, high_membership_counts = fuzzy_c_means(data, c=3, m=2, max_iter=4)
plot_clusters_with_colorful_squares(data, labels_fcm_20, centers_fcm_4, "Fuzzy C-Means (4 iteracje)")

print("K-means (Euklidesowa, 4 iteracje):")
print("Środki po 4 iteracjach:", centers_kmeans_euc_4)
print("Środki po 10 iteracjach:", centers_kmeans_euc_10)

print("K-means (Manhattan, x1 i x2, 4 iteracje):")
print("Środki po 4 iteracjach:", centers_kmeans_x1x2_4)
print("Środki po 10 iteracjach:", centers_kmeans_x1x2_10)

print("Fuzzy C-Means (4 iteracje):")
print("Środki po 4 iteracjach:", centers_fcm_4)

print("Fuzzy C-Means (20 iteracje):")
print("Środki po 20 iteracjach:", centers_fcm_20)
print("Liczba próbek o przynależności > 0.6:", high_membership_counts)
