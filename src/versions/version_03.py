from model.GeneticAlgorithm import GeneticAlgorithm
import numpy as np

def safe_fitness_v3(x, y):
  if x == 0:
    return 0
  else:
    exponent = - (abs(x) ** 2 + abs(y) ** 2)
    complex_z = complex(x) ** exponent
    return complex_z.real
    
def version_03(size, n_childrens, n_generations):
    try:
        v_safe_fitness_v3 = np.vectorize(safe_fitness_v3)
        fitness_v3 = lambda x, y: v_safe_fitness_v3(x, y)
        
        algorithm = GeneticAlgorithm(size=size, n_childrens=n_childrens, n_generations=n_generations, mutation=1, interval=[-2, 2], fitness=fitness_v3, for_max=True, version='03')
        algorithm.init()

    except Exception as e:
        raise f'Erro na execução da versão 03:\n{e}\n'