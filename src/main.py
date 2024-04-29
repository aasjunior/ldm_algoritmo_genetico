from versions.version_01 import version_01
from versions.version_02 import version_02
from versions.version_03 import version_03
from helpers.exception import generate_log
import traceback

def main():
    size = 20
    n_childrens = 14
    n_generations = 10

    try:
        version_01(size, n_childrens, n_generations)
        version_02(size, n_childrens, n_generations)
        version_03(size, n_childrens, n_generations)

    except Exception as e:
        print(f'Ocorreu um erro:\n{e}\nÉ possivel visualizar mais detalhes em: error_log.txt\n')
        generate_log(e, traceback.format_exc())

if __name__=='__main__':
    main()