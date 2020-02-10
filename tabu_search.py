from route_functions import *


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

def perform_tabu_search(max_tabu, max_iteration, city_coords):
    sol_initial = generate_rand_route(city_coords)
    sol_best = sol_initial
    candidate_best = sol_initial

    tabu_list = []
    tabu_list.append(sol_initial)

    iteration = 0

    while iteration < max_iteration:
        # print(iteration)
        neighbours = get_neighbours(candidate_best)
        candidate_best = neighbours[0]
        for candidate in neighbours:
            if (candidate not in tabu_list) and (calc_route_length(candidate, city_coords) < calc_route_length(candidate_best, city_coords)):
                candidate_best = candidate

        if calc_route_length(candidate_best, city_coords) < calc_route_length(sol_best, city_coords):
            sol_best = candidate_best
        
        tabu_list.append(candidate_best)
        if len(tabu_list) > max_tabu:
            tabu_list.pop(0)

        iteration = iteration + 1

    return sol_best
