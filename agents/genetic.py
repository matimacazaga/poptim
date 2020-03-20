import numpy as np 
import pandas as pd 
from collections import deque
from poptim.agents.base import Agent 

eps = np.finfo(float).eps

class GeneticAlgorithm:

    def __init__(self, mean_returns, sigma_returns, pop_size=500, generations=100, mutation_prob = 0.25, crossover_prob=0.75, tournament_contestants = 25, max_weight=1.):
        
        self.mean_returns = mean_returns
        self.sigma_returns = sigma_returns  
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.tournament_contestants = tournament_contestants
        self.max_weight = max_weight

    def create_population(self):
        pop = []
        for _ in range(self.pop_size):
            genome = np.random.uniform(0., 100., self.mean_returns.shape[0])
            individual = {'Genome': genome, 'Fitness': np.nan, 'ER': np.nan}
            pop.append(individual)
        return pop 

    def mutation(self, individual):
        if np.random.uniform(0., 1.) < self.mutation_prob:
            for gen in range(len(individual['Genome'])):
                individual['Genome'][gen] = individual['Genome'][gen] + np.random.normal(0,1)
                if individual['Genome'][gen] < 0.:
                    individual['Genome'][gen] = 0. 

        return individual
    
    def crossover(self, individual_1, individual_2):
        child_1 = {'Genome':[], 'Fitness':np.nan, 'ER':np.nan}
        child_2 = {'Genome':[], 'Fitness':np.nan, 'ER':np.nan}
        if np.random.uniform(0., 1.) < self.crossover_prob:
            dice = np.random.randint(1,len(individual_1['Genome']))
            child_1['Genome'] = np.hstack((individual_1['Genome'][:dice], individual_2['Genome'][dice:]))
            child_2['Genome'] = np.hstack((individual_2['Genome'][:dice], individual_1['Genome'][dice:]))
        else:
            child_1 = individual_1
            child_2 = individual_2
        return child_1, child_2

    def tournament_selection(self, pop):
        tournament = []
        for i in range(self.tournament_contestants):
            dice = np.random.randint(0, len(pop))
            already_selected = False 
            for selected in tournament:
                if np.array_equal(selected['Genome'], pop[dice]['Genome']):
                    already_selected = True 
                    i -= 1
                    break
            if already_selected == False:
                tournament.append(pop[dice])
        fitnesses = []
        for i in range(len(tournament)):
            fitnesses.append(tournament[i]['Fitness'])

        best_fit = tournament[np.argmax(fitnesses)]
        return best_fit 
    
    def evaluation_function(self, individual):
        """

        Parámetros
        ==========

        individual: 

        Retorno
        =======

        individual
        """

        constrain = False
        norm_w = individual['Genome']/individual['Genome'].sum()
        if norm_w.sum() == 1.0:
            if np.any(norm_w > self.max_weight):
                fitness = 0.
                mu_portfolio = 0.
                constrain = True 
            if constrain == False:
                mu_portfolio = np.dot(self.mean_returns,norm_w)
                sigma_portfolio = np.sqrt(np.dot(np.dot(norm_w, self.sigma_returns), norm_w))
                fitness = mu_portfolio / (sigma_portfolio + eps) #sharpe
        else:
            fitness = 0. 
            mu_portfolio = 0. 

        individual['Fitness'] = fitness 
        individual['ER'] = mu_portfolio * 100. 

        return individual 
    
    def evolve(self):
        pop = self.create_population()
        hof = {'Genome':[], 'Fitness':np.nan}
        for _ in range(self.generations):
            for i in range(len(pop)):
                pop[i] = self.evaluation_function(pop[i])
            
            fitnesses = []
            for i in range(len(pop)):
                fitnesses.append(pop[i]['Fitness'])
            
            best_in_generation = pop[np.argmax(fitnesses)]
            
            if best_in_generation['Fitness'] > hof['Fitness'] or np.isnan(hof['Fitness']):
                hof = best_in_generation
            
            selected = []
            for i in range(self.pop_size):
                selected.append(self.tournament_selection(pop))

            np.random.shuffle(selected)
            selected_A = selected[:int(len(selected)/2)]
            selected_B = selected[int(len(selected)/2):]
            next_generation = []
            for i in range(len(selected_A)):
                child_1, child_2 = self.crossover(selected_A[i], selected_B[i])
                next_generation.append(child_1)
                next_generation.append(child_2)

            for i in range(len(next_generation)):
                next_generation[i] = self.mutation(next_generation[i])
            
            pop = next_generation
        
        norm_genome = np.array(hof['Genome'])/np.array(hof['Genome']).sum() # Best Portfolio Weights
        return [hof, pop, norm_genome] 

class GeneticAgent(Agent):
    _id = 'geneticagent'
    def __init__(self, action_space, window=150, *args):
        
        self.action_space = action_space 
        self.memory = deque(maxlen=window)
        self.w = self.action_space.sample()
        
    def observe(self, observation, action, reward, done, next_observation):

        self.memory.append(observation.values)

    def act(self, observation):

        memory = np.array(self.memory)

        M = len(observation)

        mu = np.mean(memory, axis=0)

        if len(self.memory) != self.memory.maxlen:

            sigma = np.eye(M)

        else: 
            
            sigma = np.cov(memory.T)

        genetic_algo = GeneticAlgorithm(mu, sigma)
        
        self.w = genetic_algo.evolve()[2]

        return self.w 