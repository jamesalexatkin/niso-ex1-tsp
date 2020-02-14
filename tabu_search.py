from route_functions import *
from tqdm import tqdm
import statistics


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

# def get_neighbours_2opt(solution):
#     max_pos = len(solution) - 1
#     neighbours = []

#     for i in range(0, max_pos):
#         for j in range(0, max_pos):
#             # Make sure we only swap different cities
#             if i != j:
#                 new_neighbour = solution.copy()
#                 # Python slice operator is exclusive on the upper bound so the calculations are slightly different
#                 # Items 1 - (i-1) get added in order
#                 for city in solution[0 : i]:
#                     new_neighbour.append(city)
#                 # Items i - j get added in reversed order
#                 for city in reversed(solution[i : (j+1)]):
#                     new_neighbour.append(city)
#                 # Items j+1 - end get added in order
#                 for city in solution[(j+1) :]:
#                     new_neighbour.append(city)

#                 if new_neighbour not in neighbours:
#                     neighbours.append(new_neighbour)

#     return neighbours


def perform_tabu_search(max_tabu, tabu_penalty, max_iteration, city_coords):
    # Randomly generate a route to start
    sol_initial = generate_rand_route(city_coords)
    sol_best = sol_initial
    length_best = float("inf")

    tabu_list = []
    tabu_list.append(sol_initial)

    candidate_best = sol_initial

    # Track when the last time the best solution changed was
    last_iteration_changed = 0
    # If no changes to the best solution are found after this many iterations, we assume convergence and give up to save time
    STOP_AFTER = 300

    for i in tqdm(range(0, max_iteration), miniters=10):
        if i - last_iteration_changed >= STOP_AFTER:
            print(" No changes found in the past " + str(STOP_AFTER) + " iterations, exiting")
            break
        # Find neighbour solutions for the best candidate
        neighbours = get_neighbours(candidate_best)
        # Start by considering first neighbour
        candidate_best = neighbours[0]
        candidate_best_length = calc_route_length(candidate_best, city_coords)
        for candidate in neighbours:
            candidate_length = calc_route_length(candidate, city_coords)
            # Only consider better routes not in the tabu list
            # We add the new candidate in two situations:
            # 1. It's not tabu and is better
            # 2. It is tabu, but even with the penalty applied it's still better
            if ((candidate not in tabu_list) and (candidate_length < candidate_best_length)) or ((candidate in tabu_list) and (candidate_length + tabu_penalty < candidate_best_length)):
                candidate_best = candidate
                candidate_best_length = candidate_length

        # If the best candidate is better than the overall best route, that's now the best route
        if candidate_best_length < length_best:
            sol_best = candidate_best
            length_best = candidate_best_length
            last_iteration_changed = i
        
        # Last candidate considered is now tabu
        tabu_list.append(candidate_best)
        # Remove oldest tabu route if we have too many tabus
        if len(tabu_list) > max_tabu:
            tabu_list.pop(0)

        # Print best length every 10 iterations
        if i % 10 == 0:
            tqdm.write(" Best length : " + str(length_best))

    return sol_best


def tune_parameters_ts(min_tabu_size, tabu_size_step, tabu_size_iter, min_tabu_penalty, tabu_penalty_step, tabu_penalty_iter, max_iteration, city_coords):
    tabu_size = min_tabu_size

    results = []

    # Loop over tabu sizes
    for i in tqdm(range(0, tabu_size_iter)):
        tabu_penalty = min_tabu_penalty
        # Look over tabu penalties
        for j in tqdm(range(0, tabu_penalty_iter)):
            solution = perform_tabu_search(tabu_size, tabu_penalty, max_iteration, city_coords)
            length = calc_route_length(solution, city_coords)
            results.append((length, tabu_size, tabu_penalty))
            tabu_penalty = tabu_penalty + tabu_penalty_step
            print("Completed run " + str(i+1) + "/" + str(tabu_size_iter) + ", " + str(j+1) + "/" + str(tabu_penalty_iter))

        tabu_size = tabu_size + tabu_size_step

    print("\n")
    return results

def find_best_parameters_ts(results):
    best_tabu_size = 0
    best_tabu_penalty = 0
    best_length = float("inf")

    for (length, tabu_size, tabu_penalty) in results:
        if length < best_length:
            best_length = length
            best_tabu = tabu_size

    return (best_length, best_tabu, best_tabu_penalty)


def do_ts_runs(tabu_size, tabu_penalty, max_iteration, max_runs, city_coords):

    solutions = []
    lengths = []
    cumulative_lengths = 0

    best_length = float("inf")
    best_solution = []

    for i in range(0, max_runs):
        solution = perform_tabu_search(tabu_size, tabu_penalty, max_iteration, city_coords)
        solutions.append(solution)
        length = calc_route_length(solution, city_coords)
        lengths.append(length)
        cumulative_lengths = cumulative_lengths + length
        if length < best_length:
            best_length = length
            best_solution = solution

    average_length = cumulative_lengths / max_runs
    standard_dev = statistics.stdev(lengths)

    return (solutions, lengths, best_solution, best_length, average_length, standard_dev)







# def get_edges(solution):
#     edges = []
#     for i in range(0, len(solution)-1):
#         edges.append((solution[i], solution[i+1]))
#     return edges

# def get_neighbours_better(solution):
#     max_pos = len(solution) - 1
#     neighbours = []

#     original_edges = get_edges(solution)

#     for i in range(0, max_pos):
#         for j in range(0, max_pos):
#             # Make sure we only swap different cities
#             if i != j:
#                 new_neighbour = solution.copy()
#                 # Swap elements i and j
#                 city_i = solution[i]
#                 new_neighbour[i] = solution[j]
#                 new_neighbour[j] = city_i

#                 new_neighbour_edges = get_edges(new_neighbour)
#                 dropped_edges = set(original_edges) - set(new_neighbour_edges)
#                 added_edges = set(new_neighbour_edges) - set(original_edges)

#                 if new_neighbour not in neighbours:
#                     neighbours.append((new_neighbour, dropped_edges, added_edges))

#     return neighbours

# def perform_tabu_search_better(tabu_size, tabu_penalty, max_iteration, city_coords):
#     # Randomly generate a route to start
#     sol_initial = generate_rand_route(city_coords)
#     sol_best = sol_initial
#     length_best = float("inf")

#     tabu_list = []
#     # tabu_list.append(sol_initial)
#     edges_initial = get_edges(sol_initial)
#     if tabu_size < len(edges_initial):
#         for i in range(0, tabu_size):
#             tabu_list.append(edges_initial[i])
#     else:
#         for i in range(0, len(edges_initial)):
#             tabu_list.append(edges_initial[i])

#     candidate_best = sol_initial

#     for i in tqdm(range(0, max_iteration)):
#         # Find neighbour solutions for the best candidate
#         neighbours = get_neighbours_better(candidate_best)
#         # Start by considering first neighbour
#         (candidate_best, dropped_edges, new_tabu_edges) = neighbours[0]
#         candidate_best_length = calc_route_length(candidate_best, city_coords)
#         for (candidate, dropped_edges, added_edges) in neighbours:
#             candidate_length = calc_route_length(candidate, city_coords)
            
#             candidate_tabu_amount = 0
#             for edge in dropped_edges:
#                 if edge in tabu_list:
#                     candidate_tabu_amount = candidate_tabu_amount + tabu_penalty

#             # Only consider better routes not in the tabu list
#             # We add the new candidate in two situations:
#             # 1. It's not tabu and is better
#             # 2. It is tabu, but even with the penalty applied it's still better
#             if candidate_length + candidate_tabu_amount < candidate_best_length:
#                 candidate_best = candidate
#                 candidate_best_length = candidate_length
#                 new_tabu_edges = added_edges.copy()

#         # If the best candidate is better than the overall best route, that's now the best route
#         if candidate_best_length < length_best:
#             sol_best = candidate_best
#             length_best = candidate_best_length
        
#         # Last candidate considered is now tabu
#         tabu_list.extend(new_tabu_edges)
#         # Remove oldest tabu route if we have too many tabus
#         if len(tabu_list) > tabu_size:
#             tabu_list.pop(0)

#         # Print best length every 10 iterations
#         if i % 10 == 0:
#             print(" Best length : " + str(length_best))

#     return sol_best








