# Genetic approach to travelling salesperson readme

# Dependencies

matplotlib

# Explanation

Using a genetic algorithm, approximate the shortest path between a set of cities.

# Mechanics

Crossover: 

    Select two parents, and create two children by crossing over the genes of the parents.
    The children may or may not be the same as the parents.

Fitness:

    The fitness of a path is the total distance traveled. A shorter path is better.

Mutate:

    A mutation occurs with a probability of `mutation_rate`.
    A mutation replaces a gene with a random gene.

# Use

Input:

    A list of cities and their coordinates.
    A mutation rate.
    A population size.
    A number of generations.

Output:

    The shortest distance traveled.
    The shortest distance traveled per generation.


