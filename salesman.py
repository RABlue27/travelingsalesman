from os import read
import os
import random
import math
import tkinter as tk # unused 
import time
import matplotlib.pyplot as plt
import csv
from datetime import datetime

# Class to solve the travelling salesperson with a genetic algorithm
class GeneticAlgorithm:
    def __init__(self, population_size, generations, mutation_rate, crossover_rate, k, elite, cities, num):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.cities = cities
        self.fitness_values = []
        self.best_path = []
        self.best_fitness = math.inf    
        self.best_path_fitness = math.inf
        # Initialize the population
        self.population = self.initial_population()
        self.k = k
        self.elite = elite
        self.best_distance = math.inf
        # Get the current math.random seed
        # create a new random seed based on the current time 
        self.seed = time.time()
        random.seed(self.seed)
        print(self.seed)
        self.num = num


    # Function to generate a random path, for use in initial population
    def generate_path(self):
        path = []
        for i in range(len(self.cities)):
            path.append(i)
        random.shuffle(path)
        return path

    # Function to generate random population, for use as initial population
    def initial_population(self):
        population = []
        for i in range(self.population_size):
            population.append(self.generate_path())
        return population
    
    # inverse of distance -> You want to maximize this
    def fitness(self, path):
        distance = 0
        for i in range(len(path) - 1):
            distance += math.sqrt((self.cities[path[i]][0] - self.cities[path[i + 1]][0]) ** 2 + (self.cities[path[i]][1] - self.cities[path[i + 1]][1]) ** 2)
        # complete the loop
        distance += math.sqrt((self.cities[path[-1]][0] - self.cities[path[0]][0]) ** 2 + (self.cities[path[-1]][1] - self.cities[path[0]][1]) ** 2)
        
        if distance > 0:
            return 1 / distance
        else:
            return 0
    # Pure distance of function -> uninversed fitness -> You want to minimize this
    def distance(self, path):
        distance = 0
        for i in range(len(path) - 1):
            distance += math.sqrt((self.cities[path[i]][0] - self.cities[path[i + 1]][0]) ** 2 + (self.cities[path[i]][1] - self.cities[path[i + 1]][1]) ** 2)
        # complete the loop
        distance += math.sqrt((self.cities[path[-1]][0] - self.cities[path[0]][0]) ** 2 + (self.cities[path[-1]][1] - self.cities[path[0]][1]) ** 2)
        
        return distance

    # Tournement selection by selecting one of k and picking the best  (sort by fitness and take the largest )
    def selection(self, population, k):
        selected = [population[random.randrange(0, len(population))] for i in range(k)]
        # Sort selected by fitness
        selected.sort(key=lambda x: self.fitness(x))
        return selected[-1]

    # Select two parents to repdroduce using tournament selection function
    def reproduction(self, population):
        p1 = self.selection(population, self.k)
        p2 = self.selection(population, self.k)
        while p1 == p2:
            p2 = self.selection(population, self.k)
        return self.crossover(p1, p2)

    # UOI crossover 
    def crossover(self, p1, p2):
        c1, c2 = [None] * len(self.cities), [None] * len(self.cities)
        # create bitmask
        bitmask = []
        arr = [0,1]
        # bitmask 
        for i in range(len(p1)):
            bitmask.append(random.choice(arr))

        for i in range(len(bitmask)):
            if bitmask[i] == 1:
                c1[i] = p1[i]
                c2[i] = p2[i]
        # missing cities
        miss1, miss2 = [], []
        # generate list
        for i in range(len(bitmask)):
            if c1.count(p2[i]) == 0:
                miss1.append(p2[i])
            if c2.count(p1[i]) == 0:
                miss2.append(p1[i])
        # fill in missing cities
        for i in range(len(bitmask)):
            if c1[i] is None:
                c1[i] = miss1.pop(0)
            if c2[i] is None:
                c2[i] = miss2.pop(0)

        return c1, c2

    # Reproduce two parents to create a child using my own crossover (probably) method
    def crossover2(self, p1, p2):
        start = random.randrange(0, len(p1) - 1)
        end = random.randrange(start + 1, len(p1))
        child1 = p1[start:end]
        child2 = p2[start:end]
        for i in range(len(p1)):
            if p1[i] not in child1:
                child1.append(p1[i])
        for i in range(len(p2)):
            if p2[i] not in child2:
                child2.append(p2[i])
        return child1, child2
        
    # Swap 2 random cities in the path
    def mutate(self, individual):
        start = random.randrange(0, len(individual) - 1)
        end = random.randrange(start + 1, len(individual))
        individual[start], individual[end] = individual[end], individual[start]
        return individual


    # Call mutate function on entire population to mutate a
    def mutate_population(self, population, rate):
        mutated = []
        for i in range(len(population)):
            if random.random() < rate:
                res = self.mutate(population[i])
                mutated.append(res)
            else:
                mutated.append(population[i])
        return mutated
    
    # Process to create the next generation
    def next_generation(self, population):
        # sort current population by fitness
        # select best individual
        new_gen = []
        for i in range(0, self.population_size - 1, 2):
            if random.random() < self.crossover_rate:
                x, y = self.reproduction(population)
                new_gen.append(x)
                new_gen.append(y)
            else:
                new_gen.append(population[i])
                new_gen.append(population[i + 1])
       
        for i in range(len(population) - len(new_gen)):
            new_gen.append(population[i])
        # mutate the new generation
        new_gen = self.mutate_population(new_gen, self.mutation_rate)
        # Code to make sure things dont break and our population remains the same 
        for i in range(len(new_gen) - self.population_size):
            new_gen.pop()
        return new_gen

    def run(self):
        mathlist = []
        newpop = self.initial_population()
        self.best_fitness = -math.inf       
        container = []
        elitelist = []


        with open('tsp' + str(self.num) + '.csv', 'a', newline='') as csvfile:
            # Write all paramaters
            writer = csv.writer(csvfile, delimiter=',')
            #  population_size, generations, mutation_rate, crossover_rate, k, elite, cities, num):
            writer.writerow([self.population_size, self.generations, self.mutation_rate, self.crossover_rate, self.k, self.elite, len(self.cities), self.num, self.seed])
            


            # Write to CSV: population_size, generations, mutation_rate, crossover_rate
          

            self.best_distance = self.distance(newpop[0])

            # Run g generations
            for g in range(self.generations):

                # Create our new pop
                newpop = self.next_generation(self.population)
                self.population = newpop

                # Replace paths in population with paths from elitelist, checking if elitelist has paths
                if len(elitelist) > 0:
                    for path in range(len(elitelist)):
                        self.population[path] = elitelist[path][1]
                # Create a list of fitness values and contain them in a container so we can save the values we need later.
                self.fitness_values = []
                for i in range(len(self.population)):
                    container.append((self.fitness(self.population[i]), self.population[i]))
                # get index of best fitness
                # Sort fitness values. Tuple is (fitness, path)
                container.sort(key=lambda x: x[0], reverse=True)
    
                # if we find a new best fitness
                if container[0][0] > self.best_fitness:
                    self.best_fitness = container[0][0]
                    self.best_path = container[0][1]
                    self.best_distance = self.distance(self.best_path)

                # Create a list of the best paths and fitnesses after the current generation
                # Tuple is (fitness, path)
                for i in range(self.elite):
                    object = (container[i][0], container[i][1])
                    elitelist.append(object)
        
                container.sort(key=lambda x: x[0], reverse=True)
                # Purge the elite list
                for i in range(len(elitelist) - self.elite):
                    elitelist.pop()
                
                container = [] 
                # So we can graph later.
                mathlist.append((g, self.best_distance))

                average_distance = 0
                for i in range(len(self.population)):
                    average_distance += self.distance(self.population[i])
                average_distance = average_distance / len(self.population)

                # every generation
                writer.writerow([g, self.best_distance, average_distance])

            writer.writerow([self.best_distance, self.best_path])

            return self.best_path
    

# Tkinter code to make a city map. disabled for now
class TkinterApp:
    def __init__(self, cities):
        self.cities = cities
        self.canvas = tk.Canvas(width=500, height=500)
        self.draw_cities()
        self.canvas.pack()

    def draw_cities(self):
        for city in self.cities:
            x = self.cities[city][0]
            y = self.cities[city][1]
            self.canvas.create_oval(x-5, y-5, x+5, y+5, fill="black")

    def draw_path(self, path):
        for i in range(len(path)):
            x1 = self.cities[path[i]][0]
            y1 = self.cities[path[i]][1]
            x2 = self.cities[path[(i + 1) % len(path)]][0]
            y2 = self.cities[path[(i + 1) % len(path)]][1]
            self.canvas.create_line(x1, y1, x2, y2, fill="red", width=2)

            
    def update(self):
        self.canvas.update()
        
    def mainloop(self):
        self.canvas.mainloop()
    
    def clear_path(self):
        self.canvas.delete("all")
        self.draw_cities()
# Class to define a city 
class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    # distance between  cities
    def distance(self, city):
        xDistance = abs(self.x - city.x)
        yDistance = abs(self.y - city.y)
        return math.sqrt((xDistance ** 2) + (yDistance ** 2))

    #  To print
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

    # to get item
    def __getitem__(self, key):
        return self.x if key == 0 else self.y

# Returns a dictonary of cities, with index and coordinates
def new_cites(n):
    cities = {}
    keys = range(n)
    for i in keys:
        city = City(random.randrange(0, 500), random.randrange(0, 500))
        cities[i] = city
    return cities

# function to read cities from a file based on filename 
def read_cities_from_file(filename):
    cities = {}
    with open(filename, 'r') as f:
        for line in f:
            # strip line
            line = line.strip()

            # if line starts with a digit
            if line[0].isdigit():
                # split line by spaces
                line = line.split()
                # get x and y coordinates
                x = float(line[1])
                y = float(line[2])
                city = City(x, y)
                # append to cities list, index starting at 1
                cities[int(line[0])-1] = city            


    return cities

cities = read_cities_from_file("cities1.txt")
cities2 = read_cities_from_file("cities2.txt")

num = 1

# Functions to graph the CSV files 
def graph_csv(filename, num):
    x = []
    a = []
    b = []
    isHead = True
    j = 1
    with open(filename, 'r') as f:
        for line in f:
            # skip the first line
            if isHead:
                isHead = False
                continue
            # if j is 152
            if j == 151:
                break
            else:
                line = line.strip()
                line = line.split(',')
                x.append(int(float(line[0])))
                a.append(float(line[1]))
                b.append(float(line[2]))
                j += 1
        x.pop()
        a.pop()
        b.pop()

    plt.plot(x, a, label="Best")
    plt.plot(x, b, label="Average")
    plt.xlabel("Generation")
    plt.ylabel("Distance")

   
    if num % 5 == 0 and num != 0:
        plt.savefig(filename + ".png")
        plt.clf()

def create_summery_table_for_latex(filename):
    isHead = True
    j = 1
    with open(filename, 'r') as f:
        for line in f:
            # skip the first line
            if isHead:
                isHead = False
                # tabular caption
                # new page
                print("\\begin{tabular}{|c|c|c|}")
                print("\\hline")
                print("Generation & Best & Average \\\\")
                print("\\hline")
                continue
            # if j is 152
            if j == 151:
                # end table
                # draw line
                print("\\hline")
                print("\\end{tabular}")
    
                break
            else:
                # Row, Best, Average
                line = line.strip()
                line = line.split(',')
                # print the row
                if j % 50 == 0:
                    print("{} & {} & {} \\\\".format(line[0], line[1], line[2]))
                j += 1
            
                
# Just run through all the files 
for filename in os.listdir(os.getcwd()):
    if filename.endswith(".csv"):
        if num % 5 == 0 and num != 0:
            create_summery_table_for_latex(filename)
        num += 1

