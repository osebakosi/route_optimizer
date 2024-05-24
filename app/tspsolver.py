import numpy as np
from numba import prange
from itertools import permutations, combinations


class TSPSolver:
    def __init__(self, graph):
        self.graph = np.array(graph, dtype=np.float64)
        self.graph = np.insert(self.graph, 0, 0.0, axis=1)
        self.graph = np.insert(self.graph, 0, 0.0, axis=0)
        self.n = self.graph.shape[0]

    @staticmethod
    def brute_force_jit(graph):
        n = graph.shape[0]
        min_path = []
        min_cost = np.inf
        nodes = np.arange(1, n)
        for perm in permutations(nodes):
            cost = (
                graph[0, perm[0]]
                + sum(graph[perm[i], perm[i + 1]] for i in range(n - 2))
                + graph[perm[-1], 0]
            )
            if cost < min_cost:
                min_cost = cost
                min_path = [0] + list(perm) + [0]
        return min_cost, min_path

    def brute_force(self):
        return self.brute_force_jit(self.graph)

    @staticmethod
    def nearest_neighbor_jit(graph):
        n = graph.shape[0]
        start = 0
        unvisited = set(range(1, n))
        path = [start]
        current = start
        total_cost = 0
        while unvisited:
            next_city = min(unvisited, key=lambda city: graph[current, city])
            total_cost += graph[current, next_city]
            current = next_city
            path.append(current)
            unvisited.remove(current)
        total_cost += graph[current, start]
        path.append(start)
        return total_cost, path

    def nearest_neighbor(self):
        return self.nearest_neighbor_jit(self.graph)

    @staticmethod
    def held_karp_jit(graph):
        n = graph.shape[0]
        C = np.full((1 << n, n), np.inf)
        parent = np.full((1 << n, n), -1, dtype=np.int32)

        for k in range(1, n):
            C[1 << k, k] = graph[0, k]

        for subset_size in range(2, n):
            for subset in combinations(range(1, n), subset_size):
                bits = sum([1 << bit for bit in subset])
                for k in subset:
                    prev_bits = bits & ~(1 << k)
                    res = np.inf
                    min_m = -1
                    for m in subset:
                        if m == k:
                            continue
                        cost = C[prev_bits, m] + graph[m, k]
                        if cost < res:
                            res = cost
                            min_m = m
                    C[bits, k] = res
                    parent[bits, k] = min_m

        bits = (1 << n) - 2
        opt = np.inf
        p = -1
        for k in range(1, n):
            cost = C[bits, k] + graph[k, 0]
            if cost < opt:
                opt = cost
                p = k

        path = [0]
        while bits:
            path.append(p)
            new_bits = bits & ~(1 << p)
            p = parent[bits, p]
            bits = new_bits

        return opt, [0] + path[::-1]

    def held_karp(self):
        return self.held_karp_jit(self.graph)

    @staticmethod
    # @njit
    def branch_and_bound_jit(graph):
        n = graph.shape[0]

        def bound(path, current_length):
            remaining = set(range(n)) - set(path)
            estimate = current_length
            if remaining:
                last = path[-1]
                estimate += min(graph[last, j] for j in remaining)
                estimate += np.min(graph[np.nonzero(graph)])
                estimate += min(graph[j, 0] for j in remaining)
            return estimate

        def search(path, current_length, best_length, best_path):
            if len(path) == n:
                current_length += graph[path[-1], path[0]]
                if current_length < best_length:
                    return current_length, path + [path[0]]
                return best_length, best_path
            for next_city in set(range(n)) - set(path):
                new_length = current_length + graph[path[-1], next_city]
                if bound(path + [next_city], new_length) < best_length:
                    best_length, best_path = search(
                        path + [next_city], new_length, best_length, best_path
                    )
            return best_length, best_path

        best_length, best_path = search([0], 0, np.inf, [])
        return best_length, best_path

    def branch_and_bound(self):
        return self.branch_and_bound_jit(self.graph)

    @staticmethod
    # @njit
    def simulated_annealing_jit(
        graph, initial_temp=1000, cooling_rate=0.995, min_temp=1
    ):
        def total_distance(path):
            return (
                sum(graph[path[i], path[i + 1]] for i in range(len(path) - 1))
                + graph[path[-1], path[0]]
            )

        def swap(path):
            new_path = path.copy()
            i, j = np.random.choice(len(path), 2, replace=False)
            new_path[i], new_path[j] = new_path[j], new_path[i]
            return new_path

        current_temp = initial_temp
        current_path = np.arange(len(graph))
        np.random.shuffle(current_path)
        current_cost = total_distance(current_path)
        best_path, best_cost = current_path.copy(), current_cost

        while current_temp > min_temp:
            new_path = swap(current_path)
            new_cost = total_distance(new_path)
            if new_path[0] != 0:
                continue
            if new_cost < current_cost or np.random.rand() < np.exp(
                (current_cost - new_cost) / current_temp
            ):
                current_path, current_cost = new_path, new_cost
                if new_cost < best_cost:
                    best_path, best_cost = new_path, new_cost
            current_temp *= cooling_rate
        return best_cost, list(best_path) + [best_path[0]]

    def simulated_annealing(self):
        return self.simulated_annealing_jit(self.graph)

    @staticmethod
    # @njit(parallel=True)
    def ant_colony_jit(
        graph, n_ants=100, n_best=20, n_iterations=100, decay=0.95, alpha=1, beta=2
    ):

        n = graph.shape[0]

        def initialize_pheromones():
            return np.full((n, n), 1 / (n * n))

        def distance(path):
            return (
                sum(graph[path[i], path[i + 1]] for i in range(n - 1))
                + graph[path[-1], path[0]]
            )

        def update_pheromones(pheromones, paths):
            for path, cost in paths:
                for i in range(n - 1):
                    pheromones[path[i], path[i + 1]] += 1.0 / cost
                pheromones[path[-1], path[0]] += 1.0 / cost

        pheromones = initialize_pheromones()
        best_cost = np.inf
        best_path = []

        for _ in range(n_iterations):
            all_paths = []
            for _ in prange(n_ants):
                path = [0]
                visited = set(path)
                for _ in range(n - 1):
                    current = path[-1]
                    probabilities = np.zeros(n)
                    for next_city in range(n):
                        if next_city not in visited:
                            if graph[current, next_city] != 0:
                                probabilities[next_city] = (
                                    pheromones[current, next_city] ** alpha
                                ) * (1 / graph[current, next_city]) ** beta
                            else:
                                probabilities[next_city] = 1
                    probabilities /= probabilities.sum()
                    next_city = np.random.choice(np.arange(n), p=probabilities)
                    path.append(next_city)
                    visited.add(next_city)
                all_paths.append((path, distance(path)))
            all_paths.sort(key=lambda x: x[1])
            for path, cost in all_paths[:n_best]:
                if cost < best_cost:
                    best_cost = cost
                    best_path = path
            pheromones *= decay
            update_pheromones(pheromones, all_paths[:n_best])
        return best_cost, list(best_path) + [best_path[0]]

    def ant_colony(self):
        return self.ant_colony_jit(self.graph)

    @staticmethod
    def genetic_algorithm_njit(
        graph, population_size=100, generations=500, mutation_rate=0.1, elite_size=20
    ):
        def create_route():
            n = graph.shape[0]
            route = np.arange(n)
            np.random.shuffle(route)
            return route

        def initial_population(population_size):
            return np.array([create_route() for _ in range(population_size)])

        def route_distance(route):
            distance = 0
            for i in range(len(route)):
                from_city = route[i]
                to_city = route[(i + 1) % len(route)]
                distance += graph[from_city, to_city]
            return distance

        def rank_routes(population):
            distances = np.array([route_distance(route) for route in population])
            return distances.argsort()

        def selection(population, ranked_population, elite_size):
            selection_results = ranked_population[:elite_size]
            df_fitness = 1 / (ranked_population + 1)
            fitness_prob = df_fitness / df_fitness.sum()
            selected_indices = np.random.choice(
                ranked_population, size=len(population) - elite_size, p=fitness_prob
            )
            return np.concatenate((selection_results, selected_indices))

        def mating_pool(population, selected_indices):
            return population[selected_indices]

        def breed(parent1, parent2):
            child = [-1] * len(parent1)
            start, end = sorted(np.random.randint(len(parent1), size=2))
            child[start:end] = parent1[start:end]

            fill_values = [item for item in parent2 if item not in child]
            fill_index = 0
            for i in range(len(child)):
                if child[i] == -1:
                    child[i] = fill_values[fill_index]
                    fill_index += 1

            return np.array(child)

        def breed_population(mating_pool, elite_size):
            children = mating_pool[:elite_size].tolist()
            length = len(mating_pool) - elite_size
            pool = np.random.permutation(mating_pool)

            for i in range(length):
                child = breed(pool[i], pool[len(mating_pool) - i - 1])
                children.append(child)

            return np.array(children)

        def mutate(route, mutation_rate):
            for swapped in range(len(route)):
                if np.random.rand() < mutation_rate:
                    swap_with = np.random.randint(len(route))

                    route[swapped], route[swap_with] = route[swap_with], route[swapped]

            return route

        def mutate_population(population, mutation_rate):
            return np.array([mutate(route, mutation_rate) for route in population])

        def next_generation(current_generation, elite_size, mutation_rate):
            ranked_population = rank_routes(current_generation)
            selected_indices = selection(
                current_generation, ranked_population, elite_size
            )
            matingpool = mating_pool(current_generation, selected_indices)
            children = breed_population(matingpool, elite_size)
            next_gen = mutate_population(children, mutation_rate)
            return next_gen

        population = initial_population(population_size)
        for i in range(generations):
            population = next_generation(population, elite_size, mutation_rate)

        best_route = [
            population[idx]
            for idx in rank_routes(population)
            if population[idx][0] == 0
        ][0]
        best_cost = route_distance(best_route)

        return best_cost, np.append(best_route, best_route[0])

    def genetic_algorithm(self):
        return self.genetic_algorithm_njit(self.graph)
