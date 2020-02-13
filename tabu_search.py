from route_functions import *
from tqdm import tqdm


def get_neighbours(solution):
    max_pos = len(solution) - 1
    neighbours = []

    for i in range(0, max_pos):
        for j in range(0, max_pos):
            # Make sure we only swap different cities
            if i != j:
                new_neighbour = solution.copy()
                # Swap elements i and j
                city_i = solution[i]
                new_neighbour[i] = solution[j]
                new_neighbour[j] = city_i

                if new_neighbour not in neighbours:
                    neighbours.append(new_neighbour)

    return neighbours

def get_neighbours_2opt(solution):
    max_pos = len(solution) - 1
    neighbours = []

    for i in range(0, max_pos):
        for j in range(0, max_pos):
            # Make sure we only swap different cities
            if i != j:
                new_neighbour = solution.copy()
                # Python slice operator is exclusive on the upper bound so the calculations are slightly different
                # Items 1 - (i-1) get added in order
                for city in solution[0 : i]:
                    new_neighbour.append(city)
                # Items i - j get added in reversed order
                for city in reversed(solution[i : (j+1)]):
                    new_neighbour.append(city)
                # Items j+1 - end get added in order
                for city in solution[(j+1) :]:
                    new_neighbour.append(city)

                if new_neighbour not in neighbours:
                    neighbours.append(new_neighbour)

    return neighbours






def perform_tabu_search(max_tabu, max_iteration, city_coords):
    # Randomly generate a route to start
    sol_initial = generate_rand_route(city_coords)
    sol_best = sol_initial
    length_best = float("inf")

    tabu_list = []
    tabu_list.append(sol_initial)

    candidate_best = sol_initial

    for i in tqdm(range(0, max_iteration)):
        # Find neighbour solutions for the best candidate
        neighbours = get_neighbours(candidate_best)
        # Start by considering first neighbour
        candidate_best = neighbours[0]
        candidate_best_length = calc_route_length(candidate_best, city_coords)
        for candidate in neighbours:
            candidate_length = calc_route_length(candidate, city_coords)
            # Only consider better routes not in the tabu list
            # If the other candidate is better, that's now the best candidate
            if (candidate not in tabu_list) and (candidate_length < candidate_best_length):
                candidate_best = candidate
                candidate_best_length = candidate_length

        # If the best candidate is better than the overall best route, that's now the best route
        if candidate_best_length < length_best:
            sol_best = candidate_best
            length_best = candidate_best_length
        
        # Last candidate considered is now tabu
        tabu_list.append(candidate_best)
        # Remove oldest tabu route if we have too many tabus
        if len(tabu_list) > max_tabu:
            tabu_list.pop(0)

        # Print best length every 10 iterations
        if i % 10 == 0:
            print(" Best length : " + str(length_best))

    return sol_best
