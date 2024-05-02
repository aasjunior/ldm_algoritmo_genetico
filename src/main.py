from versions.version_01 import version_01
from versions.version_02 import version_02
from versions.version_03 import version_03
from helpers.exception import generate_log
import numpy as np
import traceback

def main():
    # Gera um número aleatório no intervalo de 20 a 100
    size = np.random.randint(20, 101)

    # Calcula n_childrens como 70% de size
    n_childrens = int(0.7 * size)

    n_generations = 10

    readme = '../README.md'
    plot_imgs = 'docs/plot'

    try:
        version_01(size, n_childrens, n_generations)
        version_02(size, n_childrens, n_generations)
        version_03(size, n_childrens, n_generations)

        print(f'\nA imagem de cada plotagem esta sendo salva no diretório {plot_imgs}. A analise do algoritmo e seus resultados podem ser observados em: {readme}')
        print(f'Obs: No VSCode, para melhor visualização do README, usar o comando CTRL + SHIFT + v.\n')
    except Exception as e:
        print(f'Ocorreu um erro:\n{e}\nÉ possivel visualizar mais detalhes em: error_log.txt\n')
        generate_log(e, traceback.format_exc())

if __name__=='__main__':
    main()