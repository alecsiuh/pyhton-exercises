import math

import pandas as pd
import numpy as np
from inspyred import ec  # ec stands for Evolutionary Computation
from random import Random
from simanneal import Annealer

# Question 1

items = pd.read_csv('../datasets/Knapsack Items.csv')

# Simmulated annealing

class Knapsack(Annealer):
    def move(self):  # how to go from this solution to a neighbor solution
        i = np.random.randint(0, len(self.state))
        if self.state[i] == 0:
            self.state[i] = 1
        else:
            self.state[i] = 0

    def energy(self):  # calculates value of objective function of solution
        total_yield = 0
        total_weight = 0
        for i in range(0, len(self.state)):
            if self.state[i] == 1:
                total_yield += items['value'][i]
                total_weight += items['weight(gr)'][i]
        if total_weight > 750:
            total_yield = -1  # really bad value
        return total_yield*(-1)  # multiply by -1 when energy should be minimized


# init_sol = np.random.uniform(-5.12, 5.12, size=30)  # 30 randomly  generated numbers between -5.12 and 5.12
init_sol = np.zeros(len(items))
knapsack = Knapsack(init_sol)
print(knapsack.anneal())  # returns the best state and energy found


# genetic algorithm

def obj_func(solution):
    total_yield = 0
    total_weight = 0
    for i in range(0, len(solution)):
        print(solution)
        if solution[i] == 1:
            total_yield += items['value'][i]
            total_weight += items['weight(gr)'][i]
    if total_weight > 750:
        total_yield = -1  # really bad value
    return total_yield*(-1)


def generate(random=None, args=None) -> []:  # defines how solutions are created
    num_inputs = args.get('num_inputs')
    solution = np.random.randint(0, 2, size=num_inputs)
    return solution



def evaluate(candidates, args={}):  # defines how fitness values are calculated for solutions
    fitness = []
    for candidate in candidates:
        fitness.append(obj_func(candidate))
    return fitness

rand = Random()
ga = ec.GA(rand)
# population: [ec.Individual] = ga.evolve(
#     generator=generate,
#     evaluator=evaluate,
#     selector=ec.selectors.tournament_selection,  # optional
#     pop_size=100,
#     maximize=False,
#     bounder=ec.Bounder(lower_bound=0, upper_bound=2000),
#     max_evaluations=10000,
#     mutation_rate=0.25,
#     num_inputs=len(items)
# )
# population.sort(reverse=True)
# print(population[0])


# Question 2

class GutterProb(Annealer):
    def move(self):  # how to go from this solution to a neighbor solution
        i = np.random.randint(0, len(self.state))
        factor = (-4 * np.random.random((1,)) + 2)[0]
        h1 = self.state[0]
        w1 = self.state[1]
        self.state[i] += factor
        h2 = self.state[0]
        w2 = self.state[1]
        if h2 + w2 + h2 > 100 or h2 < 0 or w2 < 0:
            self.state[0] = h1
            self.state[1] = w1
            self.state[i] -= factor


    def energy(self):  # calculates value of objective function of solution
        h = self.state[0]
        w = self.state[1]
        crosssec = 0
        if h + w + h > 100 or h < 0 or w < 0:
            crosssec = -1  # bad
        else:
            crosssec = (h * w)/(10000)  # convert to m2
        return crosssec*-1
h = -1
w = -1
while h < 0 or w < 0:
    h = 100 * np.random.random_sample((1,))[0]
    w = 100 - 2*h
init_sol = [h, w]
# gutter = GutterProb(init_sol)
# print(gutter.anneal())  # returns the best state and energy found

# genetic algorithm

def obj_func(solution):
    h = solution[0]
    w = solution[1]
    crosssec = 0
    if h + w + h > 100 or h < 0 or w < 0:
        crosssec = -1  # bad
    else:
        crosssec = (h * w)/(10000)  # convert to m2
    return crosssec


def generate(random=None, args=None) -> []:  # defines how solutions are created
    h = -1
    w = -1
    while h < 0 or w < 0 or h+w+h > 100:
        h = 100 * np.random.random_sample((1,))[0]
        w = 100 * np.random.random_sample((1,))[0]
    return [h, w]



def evaluate(candidates, args={}):  # defines how fitness values are calculated for solutions
    fitness = []
    for candidate in candidates:
        fitness.append(obj_func(candidate))
    return fitness

# rand = Random()
# ga = ec.GA(rand)
# population: [ec.Individual] = ga.evolve(
#     generator=generate,
#     evaluator=evaluate,
#     selector=ec.selectors.tournament_selection,  # optional
#     pop_size=100,
#     maximize=False,
#     bounder=ec.Bounder(lower_bound=0, upper_bound=2000),
#     max_evaluations=1000000,
#     mutation_rate=0.25,
#     num_inputs=2
# )
# population.sort(reverse=False)
# print(population[0])



# Question 3

# simmulated annealing
class Football(Annealer):
    def move(self):  # how to go from this solution to a neighbor solution
        i = np.random.randint(0, len(self.state))
        self.state[i] += -4 * np.random.random(1)[0] + 2
        if self.state[i] < 0:
            self.state[i] *= -1

        if i == 0:  # b
            self.state[1] = -0.5*self.state[0]*math.pi+200
        else:  # l
            self.state[0] = (-2*self.state[1]+400)/math.pi

    def energy(self):  # calculates value of objective function of solution
        b = self.state[0]
        l = self.state[1]
        circumference = 2*math.pi*(b/2) + 2*l
        if circumference != 400 or b < 0 or l < 0:
            energy = -1
        else:
            energy = b*l
        return energy*-1


# init_sol = np.random.uniform(-5.12, 5.12, size=30)  # 30 randomly generated numbers between -5.12 and 5.12
b = 0
l = 0
while (2*math.pi*(b/2) + 2*l) != 400:
    b = (200 * np.random.random_sample((1,)))[0]
    l = -0.5*b*math.pi+200

init_sol = [b, l]
# football = Football(init_sol)
# print(football.anneal())  # returns the best state and energy found


# genetic algorithm

def obj_func(solution):
    b = solution[0]
    l = solution[1]
    circumference = 2*math.pi*(b/2) + 2*l
    if circumference != 400 or b < 0 or l < 0:
        energy = -1
    else:
        energy = b*l
    return energy


def generate(random=None, args=None) -> []:  # defines how solutions are created
    b = 0
    l = 0
    while (2*math.pi*(b/2) + 2*l) != 400:
        b = (200 * np.random.random_sample((1,)))[0]
        l = -0.5*b*math.pi+200
    return [b, l]



def evaluate(candidates, args={}):  # defines how fitness values are calculated for solutions
    fitness = []
    for candidate in candidates:
        fitness.append(obj_func(candidate))
    return fitness

rand = Random()
# ga = ec.GA(rand)
# population: [ec.Individual] = ga.evolve(
#     generator=generate,
#     evaluator=evaluate,
#     selector=ec.selectors.tournament_selection,  # optional
#     pop_size=100,
#     maximize=False,
#     bounder=ec.Bounder(lower_bound=0, upper_bound=2000),
#     max_evaluations=1000000,
#     mutation_rate=0.25,
#     num_inputs=2
# )
# population.sort(reverse=False)
# print(population[0])


# Question 4

# simulated annealing

class Optimalization(Annealer):
    def move(self):  # how to go from this solution to a neighbor solution
        i = np.random.randint(0, len(self.state))
        factor = (2*np.random.random((1,))-1)[0]
        if self.state[i]+factor < -1 or self.state[i]+factor > 1:
            self.state[i] -= factor
        else:
            self.state[i] += factor


    def energy(self):  # calculates value of objective function of solution
        x1 = self.state[0]
        x2 = self.state[1]
        target = 0.2 + x1**2 + x2**2 - 0.1*math.cos(6*math.pi*x1)-0.1*math.cos(6*math.pi*x2)
        if x1 < -1 or x2 > 1 or x2 < -1 or x2 > 1:
            target = -1
        return target*-1

# init_sol = [0, 0]
# optimalization = Optimalization(init_sol)
# print(optimalization.anneal())  # returns the best state and energy found



# genetic algorithm

def obj_func(solution):
    x1 = solution[0]
    x2 = solution[1]
    target = 0.2 + x1**2 + x2**2 - 0.1*math.cos(6*math.pi*x1)-0.1*math.cos(6*math.pi*x2)
    if x1 < -1 or x2 > 1 or x2 < -1 or x2 > 1:
        target = -1
    return target


def generate(random=None, args=None) -> []:  # defines how solutions are created
    return 2*np.random.random((2,))-1



def evaluate(candidates, args={}):  # defines how fitness values are calculated for solutions
    fitness = []
    for candidate in candidates:
        fitness.append(obj_func(candidate))
    return fitness

rand = Random()
ga = ec.GA(rand)
population: [ec.Individual] = ga.evolve(
    generator=generate,
    evaluator=evaluate,
    selector=ec.selectors.tournament_selection,  # optional
    pop_size=100,
    maximize=False,
    bounder=ec.Bounder(lower_bound=0, upper_bound=2000),
    max_evaluations=1000000,
    mutation_rate=0.25,
    num_inputs=2
)
population.sort(reverse=False)
print(population[0])

# 5
#a) No
#b) In theory yes but it might take a long time and there's no guarantee that we found all possible solutions
#c) so that we won't get stuck in a local minimum
#d) so that we can run it multiple times and see if there are any better solutions
#e)
# As the temperature gets lower,
# the probability of accepting a trial point with a lower obj_function decreases
