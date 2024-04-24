from model.GeneticAlgorithm import GeneticAlgorithm
import numpy as np

def modulo(x):
    return np.where(x < 0, -1 * x, x)

def calc_xi(x):
    return (x * np.sin(np.sqrt(modulo(x))))

try:
    fitness_v1 = lambda x1, x2: 837.9658 - calc_xi(x1) - calc_xi(x2)

    algorithm = GeneticAlgorithm(size=20, n_childrens=14, n_generations=10, mutation=1, interval=[-500, 500], fitness=fitness_v1, for_max=False)
    algorithm.init()

except Exception as e:
    print(f'Ocorreu um erro:\n{e}')