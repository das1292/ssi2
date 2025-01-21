import numpy as np

def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def chebyshev_distance(p1, p2):
    return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_dissimilarity(bitmap_a, bitmap_b, distance_func):
    dissimilarity = 0
    points_a = [(i, j) for i in range(bitmap_a.shape[0]) for j in range(bitmap_a.shape[1]) if bitmap_a[i, j] == 1]
    points_b = [(i, j) for i in range(bitmap_b.shape[0]) for j in range(bitmap_b.shape[1]) if bitmap_b[i, j] == 1]

    for point_a in points_a:
        min_distance = float('inf')
        for point_b in points_b:
            dist = distance_func(point_a, point_b)
            min_distance = min(min_distance, dist)
        dissimilarity += min_distance

    return dissimilarity

def bidirectional_similarity(bitmap_a, bitmap_b, distance_func):
    dissimilarity_ab = calculate_dissimilarity(bitmap_a, bitmap_b, distance_func)
    dissimilarity_ba = calculate_dissimilarity(bitmap_b, bitmap_a, distance_func)
    return -(dissimilarity_ab + dissimilarity_ba)

znaki_wz = [
    np.array([
        [0, 0, 0, 1],
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1]
    ]),
    np.array([
        [0, 1, 1, 1],
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 1, 1, 1]
    ]),
    np.array([
        [1, 1, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 1, 1],
        [0, 0, 0, 1],
        [1, 1, 1, 0]
    ])
]

znaki_tst = [
    np.array([
        [0, 0, 0, 0],
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1]
    ])
]

results = []
for i, wz in enumerate(znaki_wz):
    for metric_name, metric_func in [("Manhattan", manhattan_distance), ("Chebyshev", chebyshev_distance), ("Euclidean", euclidean_distance)]:
        similarity = bidirectional_similarity(znaki_tst[0], wz, metric_func)
        results.append((f"znaki_tst[1] vs znaki_wz[{i + 1}]", metric_name, similarity))

for result in results:
    print(f"Comparison: {result[0]}, Metric: {result[1]}, Similarity: {result[2]:.2f}")
