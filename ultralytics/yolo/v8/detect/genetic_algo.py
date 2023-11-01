import numpy as np

class GeneticOptimizer:
    def __init__(self, config_template, population_size=100, mutation_rate=0.01, crossover_rate=0.7, generations=1000, tolerance=1e-6):
        self.config_template = config_template
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.tolerance = tolerance

    def _initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = {}
            for key, value in self.config_template.items():
                if isinstance(value, tuple) and len(value) == 2:
                    low, high = value
                    if isinstance(low, float) or isinstance(high, float):
                        individual[key] = np.random.uniform(low, high)
                    else:
                        individual[key] = np.random.randint(low, high)
                else:
                    individual[key] = value  # For non-tuple items like 'REID_CKPT'
            population.append(individual)
        return population
    def _crossover(self, parent1, parent2):
        child = {}
        for key in parent1.keys():
            if np.random.rand() > 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        return child

    def _mutate(self, child):
        for key, (low, high) in self.config_template.items():
            if np.random.rand() < self.mutation_rate:
                if isinstance(low, int) and isinstance(high, int):
                    child[key] = np.random.randint(low, high)
                else:
                    child[key] = np.random.uniform(low, high)
        return child

    def _select_parents(self, population, fitnesses):
        idx = np.argsort(fitnesses)
        sorted_population = np.array(population)[idx]
        return sorted_population[:2]

    def optimize(self, function_to_optimize):
        population = self._initialize_population()

        for generation in range(self.generations):
            fitnesses = np.array([function_to_optimize(individual) for individual in population])
            new_population = []

            for _ in range(self.population_size // 2):
                parent1, parent2 = self._select_parents(population, fitnesses)
                child1 = self._crossover(parent1, parent2)
                child2 = self._crossover(parent2, parent1)
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                new_population.extend([child1, child2])

            population = new_population

            if np.std(fitnesses) < self.tolerance:
                break

        fitnesses = np.array([function_to_optimize(individual) for individual in population])
        best_index = np.argmin(fitnesses)
        best_solution = population[best_index]
        best_fitness = fitnesses[best_index]

        return best_solution, best_fitness
