# This Genetic Algorithm tries to find a solution where the weighted sum of four variables (genes) 
# gets as close as possible to the target value of 30. It does this by evolving a population 
# through selection, crossover, and mutation over multiple generations.
import random

class GeneticAlgorithm:
    def __init__(self):
        self.population_size = 6
        self.num_genes = 4
        self.max_value = 30
        self.num_generations = 10000
        self.mutation_rate = 0.1
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            chromosome = [random.randint(0, self.max_value) for _ in range(self.num_genes)]
            population.append(chromosome)
        return population

    def objective_function(self, chromosome):
        return abs((chromosome[0] + 2 * chromosome[1] + 3 * chromosome[2] + 4 * chromosome[3]) - 30)

    def crossover(self, parent1, parent2):
        # Perform one-point crossover
        crossover_point = random.randint(1, self.num_genes - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child

    def mutation(self, chromosome):
        # Apply mutation with a certain probability
        for i in range(self.num_genes):
            if random.random() < self.mutation_rate:
                chromosome[i] = random.randint(0, self.max_value)
        return chromosome

    def create_new_population(self, selected_population):
        new_population = []

        # Crossover and mutation to create new chromosomes
        for i in range(self.population_size // 2):
            parent1 = random.choice(selected_population)
            parent2 = random.choice(selected_population)

            child = self.crossover(parent1, parent2)
            child = self.mutation(child)

            new_population.append(child)

        return new_population

    def run_genetic_algorithm(self):
        for generation in range(self.num_generations):
            # Evaluate fitness for the current population
            fitness = [self.objective_function(chromosome) for chromosome in self.population]

            # Selection: Choose the top half of the population based on fitness
            selected_indices = sorted(range(len(fitness)), key=lambda k: fitness[k])[:self.population_size // 2]
            selected_population = [self.population[i] for i in selected_indices]

            # Create new population through crossover and mutation
            new_population = self.create_new_population(selected_population)

            # Combine selected population and new population
            self.population = selected_population + new_population

        # Find the best solution
        best_solution = min(self.population, key=lambda chromosome: self.objective_function(chromosome))
        best_fitness = self.objective_function(best_solution)

        return best_solution, best_fitness

# Example usage
genetic_algo = GeneticAlgorithm()
best_solution, best_fitness = genetic_algo.run_genetic_algorithm()

print("Best solution:", best_solution)
print("Best fitness:", best_fitness)
