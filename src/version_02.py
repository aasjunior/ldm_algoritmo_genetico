from model.GeneticAlgorithm import GeneticAlgorithm
import numpy as np

try:
    fitness_v2 = lambda x, y: 20 + (x**2) + (y**2) - 10 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y))

    algorithm = GeneticAlgorithm(size=20, n_childrens=14, n_generations=10, mutation=1, interval=[-5, 5], fitness=fitness_v2, for_max=False)
    algorithm.init()

except Exception as e:
    print(f'Ocorreu um erro:\n{e}')