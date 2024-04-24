from model.GeneticAlgorithm import GeneticAlgorithm
import numpy as np

def calc_v3(x, y):
    x = np.float64(x)
    y = np.float64(y)
    if np.isscalar(x) and np.isscalar(y):
        if x != 0 and y != 0:
            return np.power(x, -((x**2)+(y**2)))
        else:
            return 0
    else:
        result = np.full_like(x, 0)
        mask = (x != 0) & (y != 0)
        # Add a small constant to the exponent to avoid large negative values
        result[mask] = np.where(x[mask] != 0, 1 / np.power(x[mask], ((x[mask]**2)+(y[mask]**2)) + 1e-9), 0)
        return result
    
try:
    fitness_v3 = lambda x, y: calc_v3(x, y)
    
    algorithm = GeneticAlgorithm(size=20, n_childrens=14, n_generations=10, mutation=1, interval=[-2, 2], fitness=fitness_v3, for_max=True)
    algorithm.init()

except Exception as e:
    print(f'Ocorreu um erro:\n{e}')