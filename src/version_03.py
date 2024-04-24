from model.GeneticAlgorithm import GeneticAlgorithm
import numpy as np

def safe_fitness_v3(x, y):
    if x == 0:
        return 0
    else:
        return np.power(float(x), -((x**2) + (y**2)))
    
fitness_v3 = np.vectorize(safe_fitness_v3)
    
algorithm = GeneticAlgorithm(size=20, n_childrens=14, n_generations=10, mutation=1, interval=[-2, 2], fitness=fitness_v3, for_max=True)
algorithm.init()
    
# try:
#     fitness_v3 = np.vectorize(safe_fitness_v3)
    
#     algorithm = GeneticAlgorithm(size=20, n_childrens=14, n_generations=10, mutation=1, interval=[-2, 2], fitness=fitness_v3, for_max=True)
#     algorithm.init()

# except Exception as e:
#     print(f'Ocorreu um erro:\n{e}')