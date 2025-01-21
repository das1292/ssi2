import numpy as np


def euclidean_distance(p1, p2):
    return round(np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2), 2)


def manhattan_distance(p1, p2):
    return round(abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]), 2)


def chebyshev_distance(p1, p2):
    return round(max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1])), 2)


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


def display_bitmap(bitmap):
    print("=" * 20)
    print("Bitmap:")
    for row in bitmap:
        print(" ".join(str(cell) for cell in row))


def shift_bitmap(bitmap, direction, steps):
    shifted_bitmap = np.zeros_like(bitmap)
    if direction == 'L':
        for y in range(bitmap.shape[0]):
            for x in range(bitmap.shape[1]):
                shifted_bitmap[y][(x - steps) % bitmap.shape[1]] = bitmap[y][x]
    elif direction == 'R':
        for y in range(bitmap.shape[0]):
            for x in range(bitmap.shape[1]):
                shifted_bitmap[y][(x + steps) % bitmap.shape[1]] = bitmap[y][x]
    elif direction == 'U':
        for y in range(bitmap.shape[0]):
            for x in range(bitmap.shape[1]):
                shifted_bitmap[(y - steps) % bitmap.shape[0]][x] = bitmap[y][x]
    elif direction == 'D':
        for y in range(bitmap.shape[0]):
            for x in range(bitmap.shape[1]):
                shifted_bitmap[(y + steps) % bitmap.shape[0]][x] = bitmap[y][x]

    return shifted_bitmap


def generate_report(bitmap, patterns, distance_func, direction, steps):
    print("Initial bitmap:")
    display_bitmap(bitmap)

    if direction != "brak":
        bitmap = shift_bitmap(bitmap, direction, steps)
        print(f"Shifted bitmap ({direction} by {steps}):")
        display_bitmap(bitmap)

    print(f"Using distance metric: {distance_func.__name__}")
    for i, pattern in enumerate(patterns):
        similarity = bidirectional_similarity(bitmap, pattern, distance_func)
        print(f"Comparison with pattern {i + 1}:")
        print(f"Bidirectional similarity: {similarity:.2f}")
        print("-" * 40)


patterns = [
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

test_bitmaps = [
    np.array([
        [0, 0, 0, 0],
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1]
    ]),
    np.array([
        [1, 1, 1, 1],
        [0, 0, 0, 1],
        [1, 1, 1, 1],
        [0, 0, 1, 1],
        [1, 1, 1, 1]
    ]),
    np.array([
        [1, 1, 1, 1],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 1]
    ])
]

# Zad 1
# generate_report(test_bitmaps[0], patterns, manhattan_distance, "brak", 0)
# generate_report(test_bitmaps[0], patterns, euclidean_distance, "brak", 0)
# generate_report(test_bitmaps[0], patterns, chebyshev_distance, "brak", 0)

# Zad 2
# generate_report(test_bitmaps[0], patterns, euclidean_distance, "L", 1)

# Zad 3
generate_report(test_bitmaps[1], patterns, euclidean_distance, "brak", 0)
generate_report(test_bitmaps[2], patterns, euclidean_distance, "brak", 0)
