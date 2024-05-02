from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import os

class GeneticAlgorithm:
    """
    Esta classe implementa um algoritmo genético.

    Args:
        size (int): O tamanho da população (número de indivíduos).
        n_generations (int): O número de gerações para evolução.
        n_childrens (int): O número de filhos a serem gerados em cada geração.
        mutation (float): A probabilidade de mutação para cada indivíduo.
        fitness (function): A função de fitness a ser avaliada, recebendo dois argumentos (x e y) e retornando um valor de fitness.
        interval (tuple): O intervalo (mínimo, máximo) para o espaço de busca.
        for_max (bool, optional): Se deve encontrar o valor mínimo (False) ou máximo (True). Padrão para True.
        version (str, optional): Uma string de versão para identificar os resultados. Padrão para None.
    """

    def __init__(self, size, n_generations, n_childrens, mutation, fitness, interval, for_max=True, version=None, show_plot=True, save_docs=True):
        """
        Initialize the Genetic Algorithm object.

        Args:
            size (int): O tamanho da população (número de indivíduos).
            n_generations (int): O número de gerações para evolução.
            n_childrens (int): O número de filhos a serem gerados em cada geração.
            mutation (float): A probabilidade de mutação para cada indivíduo.
            fitness (function): A função de fitness a ser avaliada, recebendo dois argumentos (x e y) e retornando um valor de fitness.
            interval (tuple): O intervalo (mínimo, máximo) para o espaço de busca.
            for_max (bool, optional): Se deve encontrar o valor mínimo (False) ou máximo (True). Padrão para True.
            version (str, optional): Uma string de versão para identificar os resultados. Padrão para None.
        """
        self.size = size
        self.n_generations = n_generations
        self.n_childrens = n_childrens
        self.mutation = mutation
        self.fitness = fitness
        self.interval = interval
        self.for_max = for_max
        self.version = version

        self.population = self.init_population()
        self.childrens = []
        self.fitness_avgs = []
        self.fitness_max = []
        self.fitness_min = []

        self.results_file_path = f'docs/results_{version}.md' if version else 'docs/results.md'
        self.show = show_plot
        self.save_docs = save_docs

    def evaluate(self, x, y):
        """
        Evaluate the fitness of an individual (x, y).

        Args:
            x (float): A coordenada x do indivíduo.
            y (float): A coordenada y do indivíduo.

        Returns:
            float: O valor de fitness do indivíduo.
        """
        return self.fitness(x, y)

    def init_population(self):
        """
        Initialize the population with random individuals.

        Returns:
            list: Uma lista de indivíduos, onde cada indivíduo é uma lista contendo [x, y, fitness].
        """
        population = []
        for i in range(self.size):
            x = random.randint(self.interval[0], self.interval[1])
            y = random.randint(self.interval[0], self.interval[1])
            fitness = self.evaluate(x, y)
            individual = [x, y, fitness]
            population.append(individual)

        return population

    def select_father(self):
        """
        Select a father individual based on a tournament selection strategy.

        Returns:
            int: O índice do indivíduo pai selecionado na população.
        """
        max = len(self.population) - 1
        pos_candidate1 = random.randint(0, max)
        pos_candidate2 = random.randint(0, max)

        pos_father = 0
        
        if(self.population[pos_candidate1][2] > self.population[pos_candidate2][2]):
            pos_father = pos_candidate1
        else:
            pos_father = pos_candidate2              

        return pos_father

    def intersection(self, pos_father1, pos_father2):
        """
        Perform crossover between two father individuals to generate two children.

        Args:
            pos_father1 (int): O índice do primeiro indivíduo pai na população.
            pos_father2 (int): O índice do segundo indivíduo pai na população.

        Returns:
            list: Uma lista contendo dois filhos indivíduos, cada um representado por uma lista [x, y, fitness].
        """
        x_c1 = self.population[pos_father1][0]
        x_c2 = self.population[pos_father2][0]
        y_c1 = self.population[pos_father2][1]
        y_c2 = self.population[pos_father1][1]

        fitness_c1 = self.evaluate(x_c1, y_c1)
        fitness_c2 = self.evaluate(x_c2, y_c2)

        return [x_c1, y_c1, fitness_c1], [x_c2, y_c2, fitness_c2]
    
    def mutate(self, children):
        """
        Apply mutation to two children individuals.

        Args:
            children (list): Uma lista contendo dois filhos indivíduos, cada um representado por uma lista [x, y, fitness].

        Returns:
            list: Uma lista contendo os dois filhos indivíduos mutados, cada um representado por uma lista [x, y, fitness].
        """        
        x = random.randint(0, 100)
        y = random.randint(0, 100)

        if(x <= self.mutation):
            children[0] = random.randint(self.interval[0], self.interval[1])
        
        if(y <= self.mutation):
            children[1] = random.randint(self.interval[0], self.interval[1])

        return children
    
    def discard(self):
        """
        Discard the worst individuals from the population.
        """
        ind = 1
        for_max = not self.for_max

        while ind <= self.n_childrens:
            index = self.tournament_selection(2, for_max)
            self.population.pop(index)
            ind +=1

    def min_discard(self, individuals):
        """
        Discard the worst individuals based on minimum fitness.

        Args:
            individuals (list): Uma lista de indivíduos, onde cada indivíduo é representado por uma lista [x, y, fitness].
        """    
        self.population = sorted(individuals, key=lambda x:x[2], reverse=True)
        self.discard()
            
    def max_discard(self, individuals):
        """
        Discard the worst individuals based on maximum fitness.

        Args:
            individuals (list): Uma lista de indivíduos, onde cada indivíduo é representado por uma lista [x, y, fitness].
        """
        self.population = sorted(individuals, key=lambda x:x[2]) 
        self.discard()

    def tournament_selection(self, n, for_max=True):
        """
        Realiza uma seleção por torneio para selecionar um indivíduo.

        Args:
            n (int): O número de candidatos participantes do torneio.
            for_max (bool, optional): Se deve selecionar o indivíduo com o maior (True) ou menor (False) valor de fitness. Padrão para True.

        Returns:
            int: O índice do indivíduo selecionado na população.

        Este método implementa uma estratégia de seleção por torneio. Ele seleciona aleatoriamente `n` indivíduos da população e compara seus valores de fitness. Se `for_max` for True, o indivíduo com o maior fitness é escolhido. Caso contrário, o indivíduo com o menor fitness é selecionado. O índice do indivíduo escolhido é retornado.
        """
        candidate_index = random.sample(range(len(self.population)), n)
        #print(candidates)

        candidates = [(i, self.population[i]) for i in candidate_index]
        
        if(for_max):
            best_index, _ = max(candidates, key=lambda x: x[1][2])
        else:
            best_index, _ = min(candidates, key=lambda x: x[1][2])

        return best_index
        
    def roulette_selection(self):
        """
        Realiza uma seleção por roleta para selecionar um indivíduo.

        Returns:
            list: O indivíduo selecionado da população.

        Este método implementa uma estratégia de seleção por roleta. Ele calcula o fitness total de toda a população. Em seguida, gera um valor aleatório entre 0 e o fitness total. A população é iterada, e o valor de fitness atual é adicionado a um total acumulado. Se o total acumulado se tornar maior ou igual ao valor aleatório, o indivíduo atual é selecionado e retornado.
        """
        sum_fitness = sum(individual[2] for individual in self.population)

        select = random.uniform(0, sum_fitness)

        current = 0

        for individual in self.population:
            current += individual[2]

            if current >= select:
                return individual

    def generate(self):
        """
        Gera novos indivíduos filhos.

        Este método gera novos indivíduos filhos realizando crossover e mutação em indivíduos parentais selecionados por meio de seleção por torneio. Ele itera até que o número desejado de filhos seja gerado:

        1. Seleciona dois indivíduos parentais usando o método `tournament_selection` (chamado duas vezes com `for_max` definido como False para encontrar pais menos aptos).
        2. Realiza crossover nos pais selecionados usando o método `intersection` para gerar dois filhos.
        3. Aplica mutação a cada filho usando o método `mutate`.
        4. Adiciona os filhos mutados à lista `childrens`.

        Este processo continua até que o número necessário de filhos seja gerado.
        """
        n = 1

        while n <= self.n_childrens/2:
            pos_father1 = self.tournament_selection(2, for_max=False)
            pos_father2 = self.tournament_selection(2, for_max=False)

            children1, children2 = self.intersection(pos_father1, pos_father2)

            children1 = self.mutate(children1)
            children2 = self.mutate(children2)

            self.childrens.append(children1)
            self.childrens.append(children2)

            n += 1

    def avg_fitness(self):
        """
        Calcula o fitness médio da população.

        Returns:
            float: O valor médio de fitness.

        Este método calcula o fitness médio da população somando os valores de fitness de todos os indivíduos e dividindo pelo tamanho da população.
        """
        sum_fitness = sum(individual[2] for individual in self.population)

        return sum_fitness / len(self.population)
    
    def save_plot(self, name):
        """
        Salva o gráfico de fitness como uma imagem PNG.

        Este método tenta salvar o gráfico de fitness 3D gerado pelo método `plot_fitness` como uma imagem PNG no diretório `docs/plot`. Ele cria o diretório se ele não existir. Se ocorrerem erros durante o salvamento, ele gera uma exceção com uma mensagem informativa.
        """
        try:
            plot_dir = 'docs/plot'

            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            
            plt.savefig(f'{plot_dir}/{name}.png')
        
        except Exception as e:
            raise Exception(f'\nErro ao tentar salvar a plotagem como imagem: \n{e}\n')


    def plot_fitness(self):
        """
        Cria e exibe o gráfico de fitness 3D.

        Este método cria um gráfico 3D da função de fitness usando a biblioteca `matplotlib`. Ele faz o seguinte:

        1. Gera uma grade de pontos dentro do intervalo especificado.
        2. Avalia a função de fitness em cada ponto da grade.
        3. Cria um gráfico de superfície 3D usando os valores de fitness.
        4. Identifica o melhor indivíduo (maior ou menor fitness dependendo de `for_max`) e o plota como um ponto de dispersão azul.
        5. Define rótulos para os eixos e um título para o gráfico.
        6. Salva o gráfico como uma imagem PNG usando o método `save_plot`.
        7. Por fim, exibe o gráfico usando `plt.show`.
        """
        x = np.linspace(self.interval[0], self.interval[1], 100)
        y = np.linspace(self.interval[0], self.interval[1], 100)
        x, y = np.meshgrid(x, y)
        z = self.evaluate(x, y)

        x_pop = [ind[0] for ind in self.population]
        y_pop = [ind[1] for ind in self.population]
        z_pop = [ind[2] for ind in self.population]

        fig = plt.figure(num=f'Versão {self.version}')
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z, cmap='viridis', alpha=0.8)

        ax.scatter(x_pop, y_pop, z_pop, color='red', label='Indivíduos na população')

        if(self.for_max):
            best = max(self.population, key=lambda x: x[2])
        else:
            best = min(self.population, key=lambda x: x[2])
            
        ax.scatter(best[0], best[1], best[2], color='blue', label='O melhor indíviduo')

        print(f'\nMelhor individuo versão {self.version}:\nx: {best[0]}\ny: {best[1]}\nfitness: {best[2]}')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Fitness')
        ax.set_title('Superfície da função custo')

        ax.text(best[0], best[1], best[2], f' Melhor indivíduo:\n x={best[0]}, y={best[1]}, fitness={best[2]}', transform=ax.transAxes, verticalalignment='top', color='red')

        self.save_plot(f'plot_fitness_v{self.version}')
        
        if self.show:
            plt.show()

    def plot_evolution(self):
        """
        Plota a evolução temporal das populações e a variação dos valores de fitness médio, máximo e mínimo.

        Args:
            show (bool, optional): Se deve exibir o gráfico (True) ou apenas salvá-lo (False). Padrão para True.

        Este método gera um gráfico com três linhas:

        1. **Fitness médio:** Mostra a variação do fitness médio da população ao longo das gerações.
        2. **Fitness máximo:** Apresenta a trajetória do fitness máximo da população ao longo das gerações.
        3. **Fitness mínimo:** Apresenta a trajetória do fitness mínimo da população ao longo das gerações.

        O gráfico é salvo como uma imagem PNG no diretório `docs/plot` com o nome `plot_v{self.version}.png`.

        Se `show` for True, o gráfico também é exibido na tela.
        """
        plt.figure(num=f'Versão {self.version}')
        generations = range(len(self.fitness_avgs))  # Cria um array de números de geração
        plt.plot(generations, self.fitness_avgs, label='Média', marker='o')
        plt.plot(generations, self.fitness_max, label='Máximo', marker='o')
        plt.plot(generations, self.fitness_min, label='Mínimo', marker='o')
        
        plt.ylabel('Fitness')
        plt.xlabel('Geração')
        plt.legend()

        plt.xticks(generations)
        plt.grid(True)

        plt.title(f'Evolução da população - Versão {self.version}')

        self.save_plot(f'plot_evolution_v{self.version}')

        if self.show:
            plt.show()
            
    def plot_results(self, show=True):
        pass

    def check_individual_best(self, count_generations):
        """
        Verifica e salva o melhor indivíduo e os valores de fitness para a geração atual.

        Args:
            count_generations (int): O número da geração atual.

        Este método identifica o melhor indivíduo (maior ou menor fitness dependendo de `for_max`) na geração atual e armazena seus valores de fitness. 

        - `pos_best`: Encontra a posição do melhor indivíduo na população usando `max` ou `min` com uma função lambda para selecionar o indivíduo com o maior (ou menor) valor de fitness no índice 2 (fitness).
        - `max_fit`, `min_fit`, `avg`: Extrai o fitness máximo, mínimo e médio da população atual.
        - As listas `fitness_avgs`, `fitness_max`, `fitness_min` armazenam os valores de fitness ao longo das gerações para análise posterior.
        - O método `save_doc` é chamado para registrar os resultados da geração atual.
        """
        pos_best = len(self.population) - 1
        max_fit = max(self.population, key=lambda x:x[2])[2]
        min_fit = min(self.population, key=lambda x:x[2])[2]
        avg = self.avg_fitness()
        
        self.fitness_avgs.append(avg)
        self.fitness_max.append(max_fit)
        self.fitness_min.append(min_fit)

        if self.save_docs:
            self.save_doc(pos_best, max_fit, min_fit, avg, count_generations)

    def init_results_file(self):
        """
        Inicializa o arquivo de resultados com um cabeçalho.

        Este método cria ou sobrescreve o arquivo de resultados (`docs/results.md`) e escreve um cabeçalho nele.

        - Verifica se o arquivo existe e o remove, se necessário.
        - Lê o cabeçalho de um arquivo base (`docs/base/base.md`).
        - Adiciona a versão (se houver) ao cabeçalho e o grava no arquivo de resultados.
        """
        try:
            if os.path.exists(self.results_file_path):
                os.remove(self.results_file_path)
        
            with open('docs/base/base.md', 'r', encoding='utf-8') as file:
                header = file.read()

            header += f': Versão {self.version} \n\n' if(self.version) else '\n\n'

            with open(self.results_file_path, 'w', encoding='utf-8') as file:
                file.write(header)

        except Exception as e:
            raise Exception(f'\nErro ao tentar salvar os resultados da versão {self.version}: \n{e}\n')

    def save_doc(self, pos_best, max_fit, min_fit, avg, count_generations):
        """
        Salva a documentação da geração atual no arquivo de resultados.

        Args:
            pos_best (int): A posição do melhor indivíduo na população.
            max_fit (float): O valor de fitness máximo da geração.
            min_fit (float): O valor de fitness mínimo da geração.
            avg (float): O valor médio de fitness da geração.
            count_generations (int): O número da geração atual.

        Este método converte a população atual em um DataFrame do Pandas e o formata para HTML.
        Em seguida, constrói uma string contendo o texto Markdown para a geração atual, incluindo:
            - Título da geração
            - Tabela HTML da população
            - Tamanho da população
            - Informações do melhor indivíduo (x, y, fitness)
            - Valores de fitness máximo, mínimo e médio
            - Linha de separação

        Por fim, o método abre o arquivo de resultados em modo de appender e adiciona o texto Markdown construído.
        """
        population_table = pd.DataFrame(self.population, columns=['x', 'y', 'fitness']).to_html()
        results_md = f'<h2>{count_generations}ª Geração:</h2>' + '\n\n'
        results_md += f'<b>Tamanho da população: </b>{str(len(self.population))} <br>'
        results_md += f'<b>O melhor individuo: </b><br>'
        results_md += f'<ul><li><b>x: </b>{self.population[pos_best][0]}</li>'
        results_md += f'<li><b>y: </b>{self.population[pos_best][1]}</li>'
        results_md += f'<li><b>fitness: </b>{self.population[pos_best][2]}</li></ul>'
        results_md += f'<b>O maior fitness: </b>{max_fit}<br>'
        results_md += f'<b>O menor fitness: </b>{min_fit}<br>'
        results_md += f'<b>Média fitness: </b>{avg}<br><br>'
        results_md += f'<b>População:</b><br>{population_table}</br><hr>'

        with open(self.results_file_path, 'a', encoding='utf-8') as file:
            file.write(results_md)

    def init(self):
        """
        Executa o algoritmo genético.

        Este método é o ponto de entrada principal do algoritmo genético. Ele executa as seguintes etapas:

        1. Inicializa um contador de gerações (`count_generations`).
        2. Chama `init_results_file` para criar ou inicializar o arquivo de resultados.
        3. Entra em um loop que itera até que o número de gerações (`n_generations`) seja atingido:
            - Inicializa uma lista vazia para os filhos (`self.childrens`).
            - Chama `generate` para gerar novos filhos.
            - Combina a população atual com os filhos para formar a nova população.
            - Aplica a seleção de descarte (`max_discard` ou `min_discard`) dependendo de buscar o máximo ou mínimo fitness.
            - Chama `check_individual_best` para identificar e armazenar o melhor indivíduo e os valores de fitness da geração atual.
            - Incrementa o contador de gerações.
        4. Imprime uma mensagem informando a geração do arquivo de resultados.
        5. Chama `plot_fitness` para gerar e exibir o gráfico 3D de fitness.
        """
        count_generations = 1

        self.init_results_file()

        while count_generations <= self.n_generations:
            self.childrens = []
            self.generate()
            self.population = self.population + self.childrens
            
            if(self.for_max):
                self.max_discard(self.population)
            else:
                self.min_discard(self.population)

            self.check_individual_best(count_generations)
            count_generations += 1

        if self.save_docs:
            print(f'Gerado arquivo com resultados em: {self.results_file_path}\n')
            self.plot_fitness()
            self.plot_evolution()