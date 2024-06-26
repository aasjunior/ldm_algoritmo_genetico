from model.GeneticAlgorithm import GeneticAlgorithm
import numpy as np

def safe_fitness_v3(x, y):
    return np.exp(x-((x**2)+(y**2)))

def version_03(size, n_childrens, n_generations, average_fitness=False):
    save_docs = not average_fitness

    try:
        fitness_v3 = np.vectorize(safe_fitness_v3)
        algorithm = GeneticAlgorithm(size=size, n_childrens=n_childrens, n_generations=n_generations, mutation=1, interval=[-2, 2], fitness=fitness_v3, for_max=True, version='03', save_docs=save_docs)
        algorithm.init()

        if average_fitness:
            return np.mean(algorithm.fitness_avgs)
        

    except Exception as e:
        raise f'Erro na execução da versão 03:\n{e}\n'