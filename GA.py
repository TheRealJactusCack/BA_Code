from collections import namedtuple
from functools import partial
from random import choices, randint, randrange, random
import time
from typing import List, Callable, Tuple

Genome = List[int]
Population = list[Genome]
FitnessFunc = Callable[[Genome], int]
PopulateFunc = Callable[[], Population]
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome], Genome]

Thing = namedtuple('Thing', ['name', 'weight', 'value'])

things = [
    Thing('Laptop', 1500, 2200),
    Thing('Headphones', 150, 250),
    Thing('Coffee Mug', 600, 50),
    Thing('Notepad', 1400, 300),
    Thing('Water Bottle', 300, 150),
    Thing('Smartphone', 10, 800),
    Thing('Tablet', 500, 600),
    Thing('Charger', 100, 100),
    Thing('Book', 5000, 400),
    Thing('Camera', 100, 1200),
]

def generate_genome(length: int) -> Genome:
    return choices([0,1], k=length)

def generate_population(size: int, genome_length: int) -> Population:
    return [generate_genome(genome_length) for _ in range(size)]

def fitness(genome: Genome, things: [Thing], weight_limit: int) -> int:
    if len(genome) != len(things):
        raise ValueError("Genome length does not match number of things")

    weight = 0
    value = 0

    for i, thing in enumerate(things):
        if genome[i] == 1:
            weight += thing.weight
            value += thing.value

        if weight > weight_limit:
            return 0

    return value

def selection_pair(population: Population, fitness_func: FitnessFunc) -> Population:
    return choices(
        population=population,
        weights=[fitness_func(genome) for genome in population],
        k=2
    )

def single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    if len(a) != len(b):
        raise ValueError("Genomes must be of the same length for crossover")
    
    length = len(a)
    if length < 2:
        return a, b  # No crossover possible

    p = randint(1, length -1)
    return a[0:p] + b[p:], b[0:p] + a[p:]

def mutation(genome: Genome, num: int = 1, probability: float = 0.5) -> Genome:
    for _ in range(num):
        index = randrange(len(genome))
        genome[index] = genome[index] if random() > probability else abs(genome[index] - 1)
    return genome


def run_evolution(
        populate_func: PopulateFunc,
        fitness_func: FitnessFunc,
        fitness_limit: int,
        selection_func: SelectionFunc = selection_pair,
        crossover_func: CrossoverFunc = single_point_crossover,
        mutation_func: MutationFunc = mutation,
        generation_limit: int = 100
    ) -> Tuple[Population, int]:
    population = populate_func()

    for i in range(generation_limit):
        population = sorted(
            population,
            key = lambda genome: fitness_func(genome),
            reverse=True
        )

        if fitness_func(population[0]) >= fitness_limit:
            break
        next_generation = population[0:2]  # Elitismus: die besten zwei Genome übernehmen

        for j in range(int(len(population) / 2) -1):
            parents = selection_func(population, fitness_func)
            offspring_a, offspring_b = crossover_func(parents[0], parents[1])
            offspring_a = mutation_func(offspring_a)
            offspring_b = mutation_func(offspring_b)
            next_generation += [offspring_a, offspring_b]
        
        population = next_generation

    population = sorted(
        population,
        key = lambda genome: fitness_func(genome),
        reverse=True
    )

    return population, i

start = time.time()

population, generations = run_evolution(
    populate_func = partial(
        generate_population, size = 10, genome_length = len(things)
    ),
    fitness_func = partial(
        fitness, things = things, weight_limit = 3000
    ),
    fitness_limit = 740,
    generation_limit = 100,
)

end = time.time() 


def genome_to_things(genome: Genome, things: [Thing]) -> [Thing]:
    result = []

    for i, thing in enumerate(things):
        if genome[i] == 1:
            result += [thing.name]
    
    return result

print(f"Generationen: {generations}")
print(f"Dauer: {time.time() - start} Sekunden")
print(f"Beste Lösung: {genome_to_things(population[0], things)}")