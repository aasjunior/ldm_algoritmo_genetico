from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import os

class GeneticAlgorithm:
    def __init__(self, size, n_generations, n_childrens, mutation, fitness, interval, for_max=True, version=None):
        self.size = size
        self.n_generations = n_generations
        self.n_childrens = n_childrens
        self.mutation = mutation
        self.fitness = fitness
        self.interval = interval
        self.for_max = for_max

        self.population = self.init_population()
        self.childrens = []
        self.fitness_avgs = []
        self.fitness_max = []
        self.fitness_min = []

        self.version = version
        self.results_file_path = f'docs/results_{version}.md' if version else 'docs/results.md'

    def evaluate(self, x, y):
        return self.fitness(x, y)

    def init_population(self):
        population = []
        for i in range(self.size):
            x = random.randint(self.interval[0], self.interval[1])
            y = random.randint(self.interval[0], self.interval[1])
            fitness = self.evaluate(x, y)
            individual = [x, y, fitness]
            population.append(individual)

        return population

    def select_father(self):
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
        x_c1 = self.population[pos_father1][0]
        x_c2 = self.population[pos_father2][0]
        y_c1 = self.population[pos_father2][1]
        y_c2 = self.population[pos_father1][1]

        fitness_c1 = self.evaluate(x_c1, y_c1)
        fitness_c2 = self.evaluate(x_c2, y_c2)

        return [x_c1, y_c1, fitness_c1], [x_c2, y_c2, fitness_c2]
    
    def mutate(self, children):
        x = random.randint(0, 100)
        y = random.randint(0, 100)

        if(x <= self.mutation):
            children[0] = random.randint(self.interval[0], self.interval[1])
        
        if(y <= self.mutation):
            children[1] = random.randint(self.interval[0], self.interval[1])

        return children
    
    def discard(self):
        ind = 1
        for_max = not self.for_max

        while ind <= self.n_childrens:
            index = self.tournament_selection(2, for_max)
            self.population.pop(index)
            ind +=1

    def min_discard(self, individuals):
        self.population = sorted(individuals, key=lambda x:x[2], reverse=True)
        self.discard()
            
    def max_discard(self, individuals):
        self.population = sorted(individuals, key=lambda x:x[2]) 
        self.discard()

    def tournament_selection(self, n, for_max=True):
        # Seleciona n índices aleatoriamente
        candidate_index = random.sample(range(len(self.population)), n)
        #print(candidates)

        candidates = [(i, self.population[i]) for i in candidate_index]
        
        if(for_max):
            best_index, _ = max(candidates, key=lambda x: x[1][2])
        else:
            best_index, _ = min(candidates, key=lambda x: x[1][2])

        return best_index
        
    def roulette_selection(self):
        sum_fitness = sum(individual[2] for individual in self.population)

        select = random.uniform(0, sum_fitness)

        current = 0

        for individual in self.population:
            current += individual[2]

            if current >= select:
                return individual

    def generate(self):
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
        sum_fitness = sum(individual[2] for individual in self.population)

        return sum_fitness / len(self.population)
    
    def save_plot(self):
        try:
            plot_dir = 'docs/plot'

            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            
            plt.savefig(f'{plot_dir}/plot_v{self.version}.png')
        
        except Exception as e:
            raise Exception(f'\nErro ao tentar salvar a plotagem como imagem: \n{e}\n')


    def plot_fitness(self):
        x = np.linspace(self.interval[0], self.interval[1], 100)
        y = np.linspace(self.interval[0], self.interval[1], 100)
        x, y = np.meshgrid(x, y)
        z = self.evaluate(x, y)

        fig = plt.figure(num=f'Versão {self.version}')
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z, cmap='viridis', alpha=0.8)

        x_coordinate = [individual[0] for individual in self.population]
        y_coordinate = [individual[1] for individual in self.population]
        costs = [individual[2] for individual in self.population]

        ax.scatter(x_coordinate, y_coordinate, costs, color='red', label='Pontos da população')

        if(self.for_max):
            best = max(self.population, key=lambda x: x[2])
        else:
            best = min(self.population, key=lambda x: x[2])
            
        ax.scatter(best[0], best[1], best[2], color='blue', label='O melhor indíviduo')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Fitness')
        ax.set_title('Superfície da função custo')

        plt.legend()

        self.save_plot()
        plt.show()

    def check_individual_best(self, count_generations):
        pos_best = len(self.population) - 1
        max_fit = max(self.population, key=lambda x:x[2])[2]
        min_fit = min(self.population, key=lambda x:x[2])[2]
        avg = self.avg_fitness()
        
        self.fitness_avgs.append(avg)
        self.fitness_max.append(max_fit)
        self.fitness_min.append(min_fit)

        self.save_doc(pos_best, max_fit, min_fit, avg, count_generations)

    def init_results_file(self):
        if os.path.exists(self.results_file_path):
            os.remove(self.results_file_path)
        
        
        with open('docs/base/base.md', 'r', encoding='utf-8') as file:
            header = file.read()

        header += f': Versão {self.version} \n\n' if(self.version) else '\n\n'

        with open(self.results_file_path, 'w', encoding='utf-8') as file:
            file.write(header)

    def save_doc(self, pos_best, max_fit, min_fit, avg, count_generations):
        population_table = pd.DataFrame(self.population, columns=['x', 'y', 'fitness']).to_html()
        results_md = f'<h2>{count_generations}ª Geração:</h2>' + '\n\n'
        results_md += population_table
        results_md += f'<b>Tamanho da população: </b>{str(len(self.population))} <br>'
        results_md += f'<b>O melhor individuo: </b><br>'
        results_md += f'<ul><li><b>x: </b>{self.population[pos_best][0]}</li>'
        results_md += f'<li><b>y: </b>{self.population[pos_best][1]}</li>'
        results_md += f'<li><b>fitness: </b>{self.population[pos_best][2]}</li></ul>'
        results_md += f'<b>O maior fitness: </b>{max_fit}<br>'
        results_md += f'<b>O menor fitness: </b>{min_fit}<br>'
        results_md += f'<b>Média fitness: </b>{avg}<br><hr>'

        with open(self.results_file_path, 'a', encoding='utf-8') as file:
            file.write(results_md)

    def init(self):
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

        print(f'Gerado arquivo com resultados em: {self.results_file_path}\n')
        self.plot_fitness()