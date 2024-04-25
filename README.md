# Trabalho 01: Algoritmos Genéticos

###### Requisitos de Software

- Python
- VSCode

### Instalação

1. Clone o repositório para o seu computador:

```
git clone https://github.com/aasjunior/ldm_algoritmo_genetico.git
```

2. Abra o projeto pelo VSCode e execute o comando pelo terminal: 

```
pip install -r requirements.txt
```

3. Navegue até o diretório `src` e execute:

```Python
python main.py
```
<br>

## Algoritmo Genético

### Versão 01

$$
\text{minimizar } z = 837,9658 -
\sum_{i=1}^2 i \cdot
\sin(\sqrt{i})
$$

```Python
from model.GeneticAlgorithm import GeneticAlgorithm
import numpy as np

def modulo(x):
    return np.where(x < 0, -1 * x, x)

def calc_xi(x):
    return (x * np.sin(np.sqrt(modulo(x))))

def version_01(size, n_childrens, n_generations):
    try:
        fitness_v1 = lambda x1, x2: 837.9658 - calc_xi(x1) - calc_xi(x2)

        algorithm = GeneticAlgorithm(size=size, n_childrens=n_childrens, n_generations=n_generations, mutation=1, interval=[-500, 500], fitness=fitness_v1, for_max=False, version='01')
        algorithm.init()

    except Exception as e:
        raise f'Erro na execução da versão 01:\n{e}\n'
```

### Versão 02

$$
\text{minimizar } z = 20 + x^2 + y^2 - 10 \cdot (\cos(2\pi x) + \cos(2\pi y))
$$

```Python
from model.GeneticAlgorithm import GeneticAlgorithm
import numpy as np

def version_02(size, n_childrens, n_generations):
    try:
        fitness_v2 = lambda x, y: 20 + (x**2) + (y**2) - 10 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y))

        algorithm = GeneticAlgorithm(size=size, n_childrens=n_childrens, n_generations=n_generations, mutation=1, interval=[-5, 5], fitness=fitness_v2, for_max=False, version='02')
        algorithm.init()

    except Exception as e:
        raise f'Erro na execução da versão 02:\n{e}\n'
```

### Versão 03

$$
\text{maximizar } z = x^{-(x ^ 2 + y ^ 2)}
$$

```Python
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
```

<br>

<hr>

###### Aviso
Este é um trabalho acadêmico realizado como tarefa da disciplina de Laboratório/Computação Natural no 5º Semestre de Desenvolvimento de Software Multiplataforma