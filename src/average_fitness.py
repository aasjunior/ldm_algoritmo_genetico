from versions.version_01 import version_01
from versions.version_02 import version_02
from versions.version_03 import version_03
from helpers.exception import generate_log
import matplotlib.pyplot as plt
import numpy as np
import traceback

def average():
    # Gera um número aleatório no intervalo de 20 a 100
    size = np.random.randint(20, 101)

    # Calcula n_childrens como 70% de size
    n_childrens = int(0.7 * size)

    n_generations = 10

    readme = '../README.md'

    fitness_avg_v01 = []
    fitness_avg_v02 = []
    fitness_avg_v03 = []

    try:
        iterations = list(range(1, n_generations + 1))

        for i in range(10):
            v01 = version_01(size, n_childrens, n_generations, average_fitness=True)
            v02 = version_02(size, n_childrens, n_generations, average_fitness=True)
            v03 = version_03(size, n_childrens, n_generations, average_fitness=True)

            fitness_avg_v01.append(v01)
            fitness_avg_v02.append(v02)
            fitness_avg_v03.append(v03)

        avg_v01 = np.mean(fitness_avg_v01)
        avg_v02 = np.mean(fitness_avg_v02)
        avg_v03 = np.mean(fitness_avg_v03)

        print("Média do fitness da versão 01 nas 10 execuções:", avg_v01)
        print("Média do fitness da versão 02 nas 10 execuções:", avg_v02)
        print("Média do fitness da versão 03 nas 10 execuções:", avg_v03)

        #plt.figure(figsize=(10, 6))
        plt.plot(iterations, fitness_avg_v01, label='Versão 01', marker='o')
        plt.plot(iterations, fitness_avg_v02, label='Versão 02', marker='o')
        plt.plot(iterations, fitness_avg_v03, label='Versão 03', marker='o')
        plt.xlabel('Iteração')
        plt.ylabel('Fitness Médio')
        plt.title('Comparação de Fitness Médio por Versão a cada Iteração')
        plt.xticks(iterations)
        plt.grid(True)
        plt.legend()
        plt.savefig('docs/plot/plot_avg_iterations.png')
        plt.show()
        
        print(f'\n{np.mean(fitness_avg_v01, axis=0)}\n')

        print(f'\nA analise do algoritmo e seus resultados podem ser observados em: {readme}')
        print(f'Obs: No VSCode, para melhor visualização do README, usar o comando CTRL + SHIFT + v.\n')
    except Exception as e:
        print(f'Ocorreu um erro:\n{e}\nÉ possivel visualizar mais detalhes em: error_log.txt\n')
        generate_log(e, traceback.format_exc())

average()