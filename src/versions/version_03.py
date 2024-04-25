from model.GeneticAlgorithm import GeneticAlgorithm
import numpy as np

def pow(n, exp):
    if n == 0:
        return 0
    else:
        return n**exp

def safe_fitness_v3(x, y):
    if x == 0:
        return 0
    else:
        return np.power(float(x), -((pow(x, 2)) + (pow(y, 2))))
    
def version_03(size, n_childrens, n_generations):
    try:
        fitness_v3 = np.vectorize(safe_fitness_v3)
        
        algorithm = GeneticAlgorithm(size=size, n_childrens=n_childrens, n_generations=n_generations, mutation=1, interval=[-2, 2], fitness=fitness_v3, for_max=True, version='03')
        algorithm.init()

    except Exception as e:
        raise f'Erro na execução da versão 03:\n{e}\n'