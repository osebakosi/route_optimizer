import numpy as np
from app.tspsolver import TSPSolver
from copy import deepcopy


class Route:
    def __init__(
        self, addresses, coords, geometry_matrix, distance_matrix, distance=None
    ):
        self.addresses = addresses
        self.coords = [([float(el) for el in coord]) for coord in coords]
        self.geometry_matrix = geometry_matrix
        self.distance_matrix = distance_matrix
        self.geometry = [
            geometry_matrix[i][i + 1] for i in range(len(geometry_matrix) - 1)
        ]
        if distance is None:
            self.distance = np.array(
                [distance_matrix[i][i + 1] for i in range(len(distance_matrix) - 1)]
            ).sum()
        else:
            self.distance = distance

    def optimize_route(self, method_name):
        def change_list(list, subs):
            return [list[idx] for idx in subs]

        def change_matrix(matrix, subs):
            new_matrix = deepcopy(matrix)
            for new_idx_row, old_idx_row in enumerate(subs):
                for new_idx_col, old_idx_col in enumerate(subs):
                    new_matrix[new_idx_row][new_idx_col] = matrix[old_idx_row][
                        old_idx_col
                    ]
            return new_matrix

        solver = TSPSolver(self.distance_matrix)
        match method_name:
            case "brute_force":
                cost, subs = solver.brute_force()
            case "nearest_neighbor":
                cost, subs = solver.nearest_neighbor()
            case "held_karp":
                cost, subs = solver.held_karp()
            case "branch_and_bound":
                cost, subs = solver.branch_and_bound()
            case "simulated_annealing":
                cost, subs = solver.simulated_annealing()
            case "ant_colony":
                cost, subs = solver.ant_colony()
            case "genetic_algorithm":
                cost, subs = solver.genetic_algorithm()

        subs = np.array(subs[1:-1]) - 1

        new_addresses = change_list(self.addresses, subs)
        new_coords = change_list(self.coords, subs)

        new_geometry_matrix = change_matrix(self.geometry_matrix, subs)
        new_distance_matrix = change_matrix(self.distance_matrix, subs)

        return Route(
            new_addresses, new_coords, new_geometry_matrix, new_distance_matrix, cost
        )
