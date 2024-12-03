from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import numpy as np

def fitness_function_1d(x):
    return np.sin(x / 10) * np.sin(x / 200)

def fitness_function_2d(x1, x2):
    return (
        np.sin(0.05 * x1) +
        np.sin(0.05 * x2) +
        0.4 * np.sin(0.15 * x1) * np.sin(0.15 * x2)
    )

def save_plot(filename):
    plt.savefig(filename)
    plt.close()

def optimize_1d(report_images):
    # Case 1: x and y values at the start, after 5, 10, 15 iterations
    scatter_x = []
    scatter_y = []
    rozrzut = 10
    wsp_przyrostu = 1.1
    l_iteracji = 15
    zakres_zmienności = (0, 100)

    np.random.seed(42)
    x = np.random.uniform(zakres_zmienności[0], zakres_zmienności[1])
    y = fitness_function_1d(x)

    case_1_results = [(0, x, fitness_function_1d(x))]
    for i in range(1, l_iteracji + 1):
        xpot = x + np.random.uniform(-rozrzut, rozrzut)
        xpot = max(0, min(xpot, 100))
        ypot = fitness_function_1d(xpot)
        if ypot >= case_1_results[-1][2]:
            x = xpot
            rozrzut *= wsp_przyrostu
        else:
            rozrzut /= wsp_przyrostu
        if i in {1, 5, 10, 15}:
            case_1_results.append((i, x, fitness_function_1d(x)))
            scatter_x.append(x)
            scatter_y.append(fitness_function_1d(x))

    plt.figure(figsize=(12, 6))
    x_vals = np.linspace(zakres_zmienności[0], zakres_zmienności[1], 1000)
    y_vals = fitness_function_1d(x_vals)
    plt.plot(x_vals, y_vals, label="Fitness function", color="blue")
    plt.scatter(scatter_x, scatter_y, color="red", label="Iteration points")
    plt.title("Optimization using Algorithm 1+1 (Case 1)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()
    image_path = "optimize_1d_case1.png"
    save_plot(image_path)
    report_images.append(("Case 1: x and y values", image_path))

    # Case 2: y and scatter values at the start and after 20 steps
    np.random.seed(43)
    x = np.random.uniform(0, 100)
    rozrzut = 10
    wsp_przyrostu = 1.1
    l_iteracji = 20

    case_2_results = [(0, fitness_function_1d(x), rozrzut)]
    for i in range(1, l_iteracji + 1):
        xpot = x + np.random.uniform(-rozrzut, rozrzut)
        xpot = max(0, min(xpot, 100))
        ypot = fitness_function_1d(xpot)
        if ypot >= case_2_results[-1][1]:
            x = xpot
            rozrzut *= wsp_przyrostu
        else:
            rozrzut /= wsp_przyrostu
        if i in {1, 20}:
            case_2_results.append((i, fitness_function_1d(x), rozrzut))

    plt.figure(figsize=(12, 6))
    x_vals = np.linspace(zakres_zmienności[0], zakres_zmienności[1], 1000)
    y_vals = fitness_function_1d(x_vals)
    plt.plot(x_vals, y_vals, label="Fitness function", color="blue")
    for i in case_2_results:
        plt.scatter(i[0], i[1], color="red", label=f"Iteration {i[0]}")
    plt.title("Optimization using Algorithm 1+1 (Case 2)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()
    image_path = "optimize_1d_case2.png"
    save_plot(image_path)
    report_images.append(("Case 2: y and scatter", image_path))

    # Case 3: x and y values for x in [15, 35] and scatter at 5
    np.random.seed(44)
    x = np.random.uniform(15, 35)
    rozrzut = 5
    wsp_przyrostu = 1.1
    l_iteracji = 20

    case_3_results = [(0, x, fitness_function_1d(x))]
    for i in range(1, l_iteracji + 1):
        xpot = x + np.random.uniform(-rozrzut, rozrzut)
        xpot = max(0, min(xpot, 100))
        ypot = fitness_function_1d(xpot)
        if ypot >= case_3_results[-1][2]:
            x = xpot
            rozrzut *= wsp_przyrostu
        else:
            rozrzut /= wsp_przyrostu
        if i in {1, 20}:
            case_3_results.append((i, x, fitness_function_1d(x)))

    plt.figure(figsize=(12, 6))
    x_vals = np.linspace(zakres_zmienności[0], zakres_zmienności[1], 1000)
    y_vals = fitness_function_1d(x_vals)
    plt.plot(x_vals, y_vals, label="Fitness function", color="blue")
    for i in case_3_results:
        plt.scatter(i[0], i[1], color="red", label=f"Iteration {i[0]}")
    plt.title("Optimization using Algorithm 1+1 (Case 3)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()
    image_path = "optimize_1d_case3.png"
    save_plot(image_path)
    report_images.append(("Case 3: x, y for x in [15, 35]", image_path))

# Task 2: Particle Swarm Optimization (PSO)
def optimize_pso(report_images):
    # Case 1: N=4, rglob=1, rinercji=0, rlok=0 - Show best and worst points before and after 5 iterations
    N = 4
    iteracje_liczba = 5
    rinercji = 0
    rglob = 1
    rlok = 0
    xmin, xmax = 0, 100
    n = 2

    np.random.seed(42)
    pozycje = np.random.uniform(xmin, xmax, (N, n))
    prędkości = np.zeros((N, n))
    lokalne_best = np.copy(pozycje)
    wartości_lokalne_best = np.array([fitness_function_2d(pozycje[i, 0], pozycje[i, 1]) for i in range(N)])
    global_best = pozycje[np.argmax(wartości_lokalne_best)]
    wartość_globalna_best = np.max(wartości_lokalne_best)

    history_positions = [np.copy(pozycje)]
    history_global_best = [global_best]

    X, Y = np.meshgrid(np.linspace(xmin, xmax, 100), np.linspace(xmin, xmax, 100))
    Z = fitness_function_2d(X, Y)

    for t in range(iteracje_liczba):
        for i in range(N):
            prędkości[i] = (
                rinercji * prędkości[i] +
                rlok * np.random.random() * (lokalne_best[i] - pozycje[i]) +
                rglob * np.random.random() * (global_best - pozycje[i])
            )

            pozycje[i] += prędkości[i]

            pozycje[i] = np.clip(pozycje[i], xmin, xmax)

            fitness = fitness_function_2d(pozycje[i, 0], pozycje[i, 1])
            if fitness > wartości_lokalne_best[i]:
                lokalne_best[i] = pozycje[i]
                wartości_lokalne_best[i] = fitness

        new_global_best = pozycje[np.argmax(wartości_lokalne_best)]
        new_global_best_value = np.max(wartości_lokalne_best)
        if new_global_best_value > wartość_globalna_best:
            global_best = new_global_best
            wartość_globalna_best = new_global_best_value

        history_positions.append(np.copy(pozycje))
        history_global_best.append(global_best)

    plt.figure(figsize=(12, 6))
    plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar()
    plt.scatter(pozycje[:, 0], pozycje[:, 1], color="red", label="Particles")
    plt.scatter(global_best[0], global_best[1], color="blue", label="Global best")
    plt.title("PSO Optimization (Case 1)")
    plt.legend()
    image_path = "pso_case1.png"
    save_plot(image_path)
    report_images.append(("Case 1: PSO with best and worst points", image_path))

    N = 20
    iteracje_liczba = 10
    rinercji = 0.5
    rglob = 2
    rlok = 1

    pozycje = np.random.uniform(xmin, xmax, (N, n))
    prędkości = np.zeros((N, n))
    lokalne_best = np.copy(pozycje)
    wartości_lokalne_best = np.array([fitness_function_2d(pozycje[i, 0], pozycje[i, 1]) for i in range(N)])
    global_best = pozycje[np.argmax(wartości_lokalne_best)]
    wartość_globalna_best = np.max(wartości_lokalne_best)

    history_positions = [np.copy(pozycje)]
    history_global_best = [global_best]

    for t in range(iteracje_liczba):
        for i in range(N):
            prędkości[i] = (
                rinercji * prędkości[i] +
                rlok * np.random.random() * (lokalne_best[i] - pozycje[i]) +
                rglob * np.random.random() * (global_best - pozycje[i])
            )

            pozycje[i] += prędkości[i]

            pozycje[i] = np.clip(pozycje[i], xmin, xmax)

            fitness = fitness_function_2d(pozycje[i, 0], pozycje[i, 1])
            if fitness > wartości_lokalne_best[i]:
                lokalne_best[i] = pozycje[i]
                wartości_lokalne_best[i] = fitness

        new_global_best = pozycje[np.argmax(wartości_lokalne_best)]
        new_global_best_value = np.max(wartości_lokalne_best)
        if new_global_best_value > wartość_globalna_best:
            global_best = new_global_best
            wartość_globalna_best = new_global_best_value

        history_positions.append(np.copy(pozycje))
        history_global_best.append(global_best)

    plt.figure(figsize=(12, 6))
    plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar()
    plt.scatter(pozycje[:, 0], pozycje[:, 1], color="red", label="Particles")
    plt.scatter(global_best[0], global_best[1], color="blue", label="Global best")
    plt.title("PSO Optimization (Case 2)")
    plt.legend()
    image_path = "pso_case2.png"
    save_plot(image_path)
    report_images.append(("Case 2: PSO with N=20", image_path))


# Zadanie 3: Strategia ewolucyjna µ + λ
def optimize_mu_lambda(report_images, turniej_rozmiar, mu, lmbda, mutacja_poziom, iteracje_liczba):
    xmin, xmax = 0, 100

    np.random.seed(42)
    populacja_rodzicielska = np.random.uniform(xmin, xmax, (mu, 2))

    # a) Położenie punktów z puli rodzicielskiej przed ewolucją
    plt.figure(figsize=(12, 8))
    plt.scatter(
        populacja_rodzicielska[:, 0], populacja_rodzicielska[:, 1],
        color='red', marker='*', label="Pula rodzicielska (przed ewolucją)", s=100
    )
    plt.title(f"Położenie punktów z puli rodzicielskiej przed ewolucją (µ={mu}, λ={lmbda})")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    image_path = f"mu_lambda_initial_{mu}_{lmbda}.png"
    save_plot(image_path)
    report_images.append(("Pula rodzicielska przed ewolucją", image_path))

    for t in range(iteracje_liczba):
        oceny_rodzicielskie = fitness_function_2d(
            populacja_rodzicielska[:, 0], populacja_rodzicielska[:, 1]
        )
        populacja_potomna = []
        for _ in range(lmbda):
            turniej_indices = np.random.choice(mu, turniej_rozmiar, replace=False)
            najlepszy_index = turniej_indices[np.argmax(oceny_rodzicielskie[turniej_indices])]
            wybrany_osobnik = populacja_rodzicielska[najlepszy_index]
            mutacja = np.random.uniform(-mutacja_poziom, mutacja_poziom, 2)
            nowy_osobnik = wybrany_osobnik + mutacja
            nowy_osobnik = np.clip(nowy_osobnik, xmin, xmax)
            populacja_potomna.append(nowy_osobnik)
        populacja_potomna = np.array(populacja_potomna)
        oceny_potomne = fitness_function_2d(
            populacja_potomna[:, 0], populacja_potomna[:, 1]
        )
        populacja_łączna = np.vstack([populacja_rodzicielska, populacja_potomna])
        oceny_łączone = np.hstack([oceny_rodzicielskie, oceny_potomne])
        najlepsze_indices = np.argsort(oceny_łączone)[-mu:]
        populacja_rodzicielska = populacja_łączna[najlepsze_indices]

        # b) Położenie punktów z puli rodzicielskiej po 3 iteracjach
        if t == 2:
            plt.figure(figsize=(12, 8))
            plt.scatter(
                populacja_rodzicielska[:, 0], populacja_rodzicielska[:, 1],
                color='red', marker='*', label=f"Pula rodzicielska po 3 iteracjach", s=100
            )
            plt.title(f"Położenie punktów z puli rodzicielskiej po 3 iteracjach (µ={mu}, λ={lmbda})")
            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.legend()
            plt.grid(True)
            image_path = f"mu_lambda_3_iterations_{mu}_{lmbda}.png"
            save_plot(image_path)
            report_images.append(("Pula rodzicielska po 3 iteracjach", image_path))

        # c) Położenie punktów z puli rodzicielskiej po 13 iteracjach
        if t == 12:
            plt.figure(figsize=(12, 8))
            plt.scatter(
                populacja_rodzicielska[:, 0], populacja_rodzicielska[:, 1],
                color='red', marker='*', label=f"Pula rodzicielska po 13 iteracjach", s=100
            )
            plt.title(f"Położenie punktów z puli rodzicielskiej po 13 iteracjach (µ={mu}, λ={lmbda})")
            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.legend()
            plt.grid(True)
            image_path = f"mu_lambda_13_iterations_{mu}_{lmbda}.png"
            save_plot(image_path)
            report_images.append(("Pula rodzicielska po 13 iteracjach", image_path))

def generate_report(report_images):
    c = canvas.Canvas("optimization_report.pdf", pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 50, "Optimization Algorithm Report")

    y_position = height - 100
    for title, image_path in report_images:
        c.setFont("Helvetica", 12)
        c.drawString(100, y_position, title)
        y_position -= 20
        c.drawImage(image_path, 100, y_position, width=400, height=300)
        y_position -= 320

    c.save()

if __name__ == "__main__":
    report_images = []
    optimize_1d(report_images)
    optimize_pso(report_images)
    optimize_mu_lambda(report_images, turniej_rozmiar=4, mu=10, lmbda=20, mutacja_poziom=5, iteracje_liczba=15)
    generate_report(report_images)
    print("Report has been generated successfully.")
