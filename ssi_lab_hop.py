import numpy as np
import matplotlib.pyplot as plt

def initialize_weights(size):
    return np.zeros((size, size))

def train_hopfield(weights, patterns):
    size = weights.shape[0]
    for pattern in patterns:
        pattern = pattern.flatten()
        weights += np.outer(pattern, pattern)
    np.fill_diagonal(weights, 0)
    return weights / size

def hopfield_update(weights, state):
    state = state.flatten()
    for i in range(len(state)):
        sum_input = np.dot(weights[i], state)
        state[i] = 1 if sum_input >= 0 else -1
    return state.reshape(int(np.sqrt(len(state))), -1)

def plot_pattern(pattern, title="Pattern"):
    plt.figure(figsize=(4, 4))
    plt.imshow(pattern, cmap='binary', interpolation='nearest')
    plt.title(title)
    plt.axis('off')
    plt.show()

def run_hopfield():
    patterns = [
        np.array([
            [1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0]
        ]),
        np.array([
            [1, 0, 0, 0, 1],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 0, 0, 0, 1]
        ]),
        np.array([
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0]
        ])
    ]

    weights = initialize_weights(25)
    weights = train_hopfield(weights, [p.flatten() * 2 - 1 for p in patterns])

    # Raport 1: Testowanie i korekta wzorców testowych
    test_patterns = [
        np.array([
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0]
        ]),
        np.array([
            [1, 1, 0, 0, 1],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [1, 1, 0, 0, 1]
        ])
    ]

    for i, test_pattern in enumerate(test_patterns):
        plot_pattern(test_pattern, title=f"Raport 1: Test Pattern {i + 1} Before Correction")
        state = test_pattern * 2 - 1
        for _ in range(5):
            state = hopfield_update(weights, state)
        corrected_pattern = (state + 1) // 2
        plot_pattern(corrected_pattern, title=f"Raport 1: Test Pattern {i + 1} After Correction")

    # Raport 2: Wyświetlanie wag pierwszego neuronu
    first_neuron_weights = weights[0].reshape(5, 5)
    print("Weights of the first neuron (5x5):")
    for row in first_neuron_weights:
        print(" ".join(f"{weight:+.2f}" for weight in row))

    # Raport 3: Negatyw pierwszego wzorca testowego
    negative_test = 1 - test_patterns[0]
    plot_pattern(negative_test, title="Raport 3: Negative Test Pattern")
    state = negative_test * 2 - 1
    for _ in range(5):
        state = hopfield_update(weights, state)
    corrected_negative = (state + 1) // 2
    plot_pattern(corrected_negative, title="Raport 3: Corrected Negative Test Pattern")

    # Raport 4: Mocno różniący się obraz i jego korekta
    different_pattern = np.random.choice([0, 1], size=(5, 5))
    plot_pattern(different_pattern, title="Raport 4: Significantly Different Pattern (Iteration 1)")
    state = different_pattern * 2 - 1
    for iteration in range(2):
        for _ in range(5):
            state = hopfield_update(weights, state)
        corrected_pattern = (state + 1) // 2
        plot_pattern(corrected_pattern, title=f"Raport 4: Corrected Pattern (Iteration {iteration + 1})")

run_hopfield()
