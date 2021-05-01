"""
First solution is using breadth first search algorithm. Each node represents a combination of working and off days.
Optimal solution is to have maximum working days, which is why the root node is the node with all of the working days
and none of the predefined off days (added trough the validation process). To shorten the time for searching,
the list of lists is built instead of graph. Each sublist contains nodes with same number of working days, so the first
found solution would be optimal.
Problem with this solution is memory usage for larger number of days and taking a lot of time to find a solution.

Second solution is the same like the first one, the difference is only that not all of the nodes are previously added.
Each time a number of working days shortens for one day, all days with less than one day are added to the queue.
This solution doesn't have the memory problem like the first one, but it is slower.

Third solution is using genetic algorithm. This is optimisation algorithm so the evaluation function is added.
This solution is by far the fastest, but the problem is that it can stuck in local optima so optimal solution
isn’t given. When building population there is a little bias towards more working days. In this algorithm
predefined off days aren't added later, so there are more states that don't meet criteria.
Evaluation function is sum of working days with high penalties if any of the conditions are not satisfied.
All hyper-parameters are randomly selected. If genetic algorithm doesn't find the solution that meets criteria,
it will try one more time before finally it returns None.

"""

import numpy as np
import datetime as dt
import itertools
from collections import defaultdict
from numpy.random import randint, rand


class Solution:
    def __init__(self, days, maxWork, minWork, maxOff, minOff, offs):
        self.days = days
        self.offs = offs
        self.weeks = days // 7
        self.max_work_days = days - len(offs)*self.weeks
        self.num_states = pow(2, self.max_work_days - 1)
        self.num_for_permu = days - len(offs)
        self.pos = self.build_possible(maxWork, minWork, maxOff, minOff)
        self.list_of_count_ones = [[] for _ in range(self.max_work_days + 1)]
        for i in range(self.num_states):
            self.list_of_count_ones[int(bin(i).count('1'))].append(i)

    # Build all possible combinations of working and off days
    def build_possible(self, maxWork, minWork, maxOff, minOff):
        possible = []
        for i in range(minWork, maxWork + 1):
            for j in range(minOff, maxOff + 1):
                possible.append((i, j))
        return possible

    # Build a boolean list from a number
    def bitfield(self, n):
        bits = [bool(int(digit)) for digit in bin(n)[2:]]
        bits = [False]*(self.max_work_days - len(bits)) + bits
        return bits

    # Add all predefined off days
    def add_rest(self, sol):
        for i in range(self.weeks):
            for j in self.offs:
                sol.insert(j + 7*i, True)
        return sol

    # Check if solution is valid (meets all criteria)
    def is_valid(self, sol):
        for i in sol:
            if not i in self.pos:
                return False
        return True

    # Get list of working and off days as tuple form list of booleans that represent working and off days
    def get_values(self, lista):
        sol = []
        ones = 0
        zeros = 0
        for i in lista:
            if not i:
                if ones == 0:
                    zeros += 1
                else:
                    sol.append((zeros, ones))
                    ones = 0
                    zeros = 1
            else:
                ones += 1
        sol.append((zeros, ones))
        return sol

    # BFS algorithm: instead of going from node to node it iterates trough the list, which represent nodes
    # in order BFS would search through them
    def solution(self):
        if self.days % 7 != 0:
            return
        for i in self.list_of_count_ones:
            for s in i:
                sol = self.bitfield(s)
                sol = self.add_rest(sol)
                sol = self.get_values(sol)
                if self.is_valid(sol):
                    return sol

class Solution2:
    def __init__(self, days, maxWork, minWork, maxOff, minOff, offs,):
        self.days = days
        self.offs = offs
        self.weeks = days // 7
        self.max_work_days = days - len(offs)*self.weeks
        self.num_states = pow(2, self.max_work_days - 1)
        self.num_for_permu = days - len(offs)
        self.pos = self.build_possible(maxWork, minWork, maxOff, minOff)
        self.node = [False] * self.max_work_days

    # Build all possible combinations of working and off days
    def build_possible(self, maxWork, minWork, maxOff, minOff):
        possible = []
        for i in range(minWork, maxWork + 1):
            for j in range(minOff, maxOff + 1):
                possible.append((i, j))
        return possible

    # Add all predefined off days
    def add_rest(self, sol):
        for i in range(self.weeks):
            for j in self.offs:
                sol.insert(j + 7*i, True)
        return sol

    # Check if solution is valid (meets all criteria)
    def is_valid(self, sol):
        for i in sol:
            if not i in self.pos:
                return False
        return True

    # Get list of working and off days as tuple form list of booleans that represent working and off days
    def get_values(self, lista):
        sol = []
        ones = 0
        zeros = 0
        for i in lista:
            if not i:
                if ones == 0:
                    zeros += 1
                else:
                    sol.append((zeros, ones))
                    ones = 0
                    zeros = 1
            else:
                ones += 1
        sol.append((zeros, ones))
        return sol

    # BFS algorithm with known order of checking nodes
    def solution(self):
        if self.days % 7 != 0:
            return
        count = -1
        queue = [self.bool2int(self.node)]
        while len(queue) != 0:
            self.node = self.int2bool(queue.pop(0), self.max_work_days)
            if sum(self.node) != count:
                count = sum(self.node)
                queue = list(set(queue))
            for i in range(len(self.node)-1, -1, -1):
                if self.node[i] == False:
                    new = self.node.copy()
                    new[i] = True
                    queue.append(self.bool2int(new))
            sol = self.add_rest(self.node)
            sol = self.get_values(sol)
            if self.is_valid(sol):
                return sol
        return None

    # Convert list of booleans to integer
    def bool2int(self, bools):
        sum = 0
        i = 0
        for b in reversed(bools):
            sum += 2**(i) * b
            i += 1
        return sum

    # Convert number to list of booleans for that number
    def int2bool(self, num, size):
        bools = []
        while num != 0:
            bools.insert(0, bool(num % 2))
            num = num // 2

        return [False]*(size-len(bools)) + bools

class Solution3:
    def __init__(self, days, maxWork, minWork, maxOff, minOff, offs, r_cross=0.9):
        self.days = days
        self.weeks = days // 7
        self.r_mut = 2 / days
        self.maxWork = maxWork
        self.minWork = minWork
        self.maxOff = maxOff
        self.minOff = minOff
        self.offs = offs
        self.all_offs = []
        for i in range(self.weeks):
            for o in offs:
                self.all_offs.append(i*7 + o)
        self.r_cross = r_cross
        self.first = True
        self.pos = self.build_possible(maxWork, minWork, maxOff, minOff)

    # Build all possible combinations of working and off days
    def build_possible(self, maxWork, minWork, maxOff, minOff):
        possible = []
        for i in range(minWork, maxWork + 1):
            for j in range(minOff, maxOff + 1):
                possible.append((i, j))
        return possible

    # Get list of working and off days as tuple form list of booleans that represent working and off days
    def get_values(self, lista):
        sol = []
        ones = 0
        zeros = 0
        for i in lista:
            if not i:
                if ones == 0:
                    zeros += 1
                else:
                    sol.append((zeros, ones))
                    ones = 0
                    zeros = 1
            else:
                ones += 1
        sol.append((zeros, ones))
        return sol

    # Calculating evaluation function
    def evaluate(self, pop):
        # False represent working days
        suma = len(pop) - sum(pop)
        sol = self.get_values(pop)
        for s in sol:
            if self.minWork > s[0] or s[0] > self.maxWork:
                suma -= 10
            if self.minOff > s[1] or s[1] > self.maxOff:
                suma -= 10
        for o in self.all_offs:
            if not pop[o]:
                suma -= 20
        return -suma

    # Method for choosing parents, returning the best out k samples
    def selection(self, k=3):
        # first random selection
        sel = randint(len(self.population))
        for i in randint(0, len(self.population), k - 1):
            # check if better (e.g. perform a tournament)
            if self.scores[i] < self.scores[sel]:
                sel = i
        return self.population[sel]

    # Function that simulates genetic material that is changed from parents to kids
    def crossover(self, p1, p2):
        # children are copies of parents by default
        c1, c2 = p1.copy(), p2.copy()
        # check for recombination
        if rand() < self.r_cross:
            # select crossover point that is not on the end of the list
            pt = randint(1, len(p1) - 2)
            # perform crossover
            c1 = p1[:pt] + p2[pt:]
            c2 = p2[:pt] + p1[pt:]
        return [c1, c2]

    # Function that simulates mutation
    def mutation(self, bitstring):
        for i in range(len(bitstring)):
            # check for a mutation
            if rand() < self.r_mut:
                # flip the bit
                bitstring[i] = not bitstring[i]
        return bitstring

    # Method that builds children in every generation
    def build_children(self):
        self.children = list()
        for i in range(0, self.n_pop, 2):
            # get selected parents in pairs
            p1, p2 = self.selected[i], self.selected[i + 1]
            # crossover and mutation
            for c in self.crossover(p1, p2):
                # mutation
                c = self.mutation(c)
                # store for next generation
                self.children.append(c)

    # Method for building first state of population
    def build_population(self, pop_size=200):
        self.n_pop = pop_size
        self.population = []
        for i in range(pop_size):
            pop = [bool(round(rand()-0.25)) for _ in range(self.days)]
            self.population.append(pop)

    # Method that calculates evaluations for every example in population
    def evaluate_scores(self):
        self.scores = []
        for i in range(len(self.population)):
            self.scores.append(self.evaluate(self.population[i]))

    # Genetic algorithm
    def genetic_algorithm(self, n_iter=600, n_pop=200):
        if self.days % 7 != 0:
            return
        # initial population of random bitstring
        self.build_population(n_pop)
        # keep track of best solution
        best, best_eval = 0, self.evaluate(self.population[0])
        # enumerate generations
        for gen in range(n_iter):
            # evaluate all candidates in the population
            self.evaluate_scores()
            # check for new best solution
            for i in range(self.n_pop):
                if self.scores[i] < best_eval:
                    best, best_eval = self.population[i], self.scores[i]
                    print(">%d, new best f(%s) = %.3f" % (gen, self.population[i], self.scores[i]))
                    print(self.get_values(self.population[i]))
            # select parents
            self.selected = [self.selection() for _ in range(self.n_pop)]
            # create the next generation
            self.build_children()
            # replace population
            self.population = self.children
        self.best = best
        self.best_eval = best_eval
        if not self.is_solution(best):
            if self.first:
                self.first = False
                best_sol = self.genetic_algorithm()
            else:
                return
            if best_sol:
                return best_sol
            else:
                return
        return self.get_values(best)

    # Check if solution is valid (meets some of the criteria)
    def is_valid(self, sol):
        for i in sol:
            if not i in self.pos:
                return False
        return True

    # Check if the best solution meets all conditions
    def is_solution(self, best):
        if not self.is_valid(self.get_values(best)):
            return False
        for o in self.all_offs:
            if not best[o]:
                return False
        return True



if __name__ == '__main__':
    offs = [5, 6]

    start = dt.datetime.now()
    s = Solution3(35, 5, 1, 2, 1, offs)
    print(s.genetic_algorithm())
    finish = dt.datetime.now()
    print(finish - start)
    offs = [2, 5]

    start = dt.datetime.now()

    print("*************************************************************")
    days = 14

    s2 = Solution2(days, 5, 1, 3, 2, offs)

    print(s2.solution())
    finish = dt.datetime.now()
    print(finish - start)

    print("********************************************************")
    start = dt.datetime.now()
    s = Solution(days, 5, 1, 3, 2, offs)
    print(s.solution())
    finish = dt.datetime.now()
    print(finish - start)
    print("********************************************************")