from model.GeneticAlgorithm import GeneticAlgorithm
import numpy as np

def version_02(size, n_childrens, n_generations, average_fitness=False):
    save_docs = False if(average_fitness) else True

    try:
        fitness_v2 = lambda x, y: 20 + (x**2) + (y**2) - 10 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y))

        algorithm = GeneticAlgorithm(size=size, n_childrens=n_childrens, n_generations=n_generations, mutation=1, interval=[-5, 5], fitness=fitness_v2, for_max=False, version='02', save_docs=save_docs)
        algorithm.init()

        if average_fitness:
            return np.mean(algorithm.fitness_avgs)

    except Exception as e:
        raise f'Erro na execução da versão 02:\n{e}\n'