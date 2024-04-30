from model.GeneticAlgorithm import GeneticAlgorithm
import numpy as np

# def safe_fitness_v3(x, y):
#   if x == 0:
#     return 0
#   else:
#     exponent = - (abs(x) ** 2 + abs(y) ** 2)
#     z = abs(float(x)) ** exponent
#     return z

def safe_fitness_v3(x, y):
    return x - (x**2 + y**2)
    

def version_03(size, n_childrens, n_generations):
    try:
        v_safe_fitness_v3 = np.vectorize(safe_fitness_v3)
        fitness_v3 = lambda x, y: v_safe_fitness_v3(x, y)
        
        algorithm = GeneticAlgorithm(size=size, n_childrens=n_childrens, n_generations=n_generations, mutation=1, interval=[-2, 2], fitness=fitness_v3, for_max=True, version='03')
        algorithm.init()

    except Exception as e:
        raise f'Erro na execução da versão 03:\n{e}\n'