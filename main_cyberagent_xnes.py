import numpy as np
from cmaes import XNES

def quadratic(x1, x2):
    return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2

def sphere(array):
    return np.sum(np.square(array))

optimizer = XNES(mean=np.zeros(10)+3, sigma=2)

for generation in range(900):
    solutions = []
    for _ in range(optimizer.population_size):
        # Ask a parameter
        x = optimizer.ask()
        # value = quadratic(x[0], x[1])
        value = sphere(x)
        solutions.append((x, value))
    values = [solution[1] for solution in solutions]
    mean_value = np.mean(np.array(values))
    print(f"#{generation} {mean_value})")
    

    # Tell evaluation values.
    optimizer.tell(solutions)