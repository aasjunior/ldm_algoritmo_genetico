from model.GeneticAlgorithm import GeneticAlgorithm
import numpy as np

def modulo(x):
    return np.where(x < 0, -1 * x, x)

def calc_xi(x):
    return (x * np.sin(np.sqrt(modulo(x))))

def version_01(size, n_childrens, n_generations, average_fitness=False):
    save_docs = not average_fitness

    try:
        fitness_v1 = lambda x1, x2: 837.9658 - calc_xi(x1) - calc_xi(x2)

        algorithm = GeneticAlgorithm(size=size, n_childrens=n_childrens, n_generations=n_generations, mutation=1, interval=[-500, 500], fitness=fitness_v1, for_max=False, version='01', save_docs=save_docs)
        algorithm.init()

        if average_fitness:
            return np.mean(algorithm.fitness_avgs)

    except Exception as e:
        raise f'Erro na execução da versão 01:\n{e}\n'