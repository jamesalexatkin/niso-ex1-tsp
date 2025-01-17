import collections
import math
import random

import matplotlib.pyplot as plt
import numpy as np

from genetic_algorithm import *
from route_functions import *
from simulated_annealing import *
from tabu_search import *

# Program start
city_coords = read_tsp_file("att48.tsp")

draw_route(read_tour_file("att48.opt.tour"), city_coords)


### SIMULATED ANNEALING
# # Finding parameters
# parameter_results = tune_parameters_sa(100, 100, 6, 2, 2, 5, 3000, city_coords)
# print(parameter_results)
# (best_length, best_temperature, best_cooling_rate) = find_best_parameters_sa(parameter_results)
# print((best_length, best_temperature, best_cooling_rate))

# # Running SA
# (solutions, lengths, best_solution, best_length, average_length, standard_dev) = do_sa_runs(best_temperature, best_cooling_rate, 3000, 30, city_coords)
# print("Average length: " + str(average_length))
# print("Standard deviation: " + str(standard_dev))
# print("Best length: " + str(best_length))
# print("Lengths:")
# print(lengths)
# draw_route(best_solution, city_coords)


### GENETIC ALGORITHM
# # Finding parameters
# parameter_results = tune_parameters_ga_nomut(1000, 1000, 5, 0.3, 0.3, 3, 0.5, 0.5, 2, 3000, city_coords)
# print(parameter_results)
# (best_length, best_pop_size, best_elitism_size, best_norm_factor) = find_best_parameters_ga(parameter_results)
# print((best_length, best_pop_size, best_elitism_size, best_norm_factor))

# # Running GA
# (solutions, lengths, best_solution, best_length, average_length, standard_dev) = do_ga_runs(best_pop_size, best_elitism_size, best_norm_factor, 9999, 3000, 30, city_coords)
# print("Parameter results: ")
# print(parameter_results)
# print("Best length, best pop size, best elite size, best norm factor: " + str(best_length) + ", " + str(best_pop_size) + ", " + str(best_elitism_size) + ", " + str(best_norm_factor))
# print("Average length: " + str(average_length))
# print("Standard deviation: " + str(standard_dev))
# print("Best length: " + str(best_length))
# print("Lengths:")
# print(lengths)
# draw_route(best_solution, city_coords)


### TABU SEARCH
# # Finding parameters
# parameter_results = tune_parameters_ts(5, 5, 5, 50, 50, 6, 3000, city_coords)
# print(parameter_results)
# (best_length, best_tabu_size, best_neighbourhood_size) = find_best_parameters_ts(parameter_results)
# print((best_length, best_tabu_size, best_neighbourhood_size))

# Running TS
# (solutions, lengths, best_solution, best_length, average_length, standard_dev) = do_ts_runs(best_tabu_size, best_neighbourhood_size, 3000, 30, city_coords)
# print("Parameter results: ")
# print(parameter_results)
# print("Best length, best tabu size, best neighbourhood size: " + str(best_length) + ", " + str(best_tabu_size) + ", " + str(best_neighbourhood_size))
# print("Average length: " + str(average_length))
# print("Standard deviation: " + str(standard_dev))
# print("Best length: " + str(best_length))
# print("Lengths:")
# print(lengths)
# draw_route(best_solution, city_coords)



# solution = perform_tabu_search(20, 1000, 3000, city_coords)
# print(solution)
# print(calc_route_length(solution, city_coords))

# solution = perform_genetic_algorithm(1500, 0.5, 2, 0.5, 3000, city_coords)
# print(solution)
# print(calc_route_length(solution, city_coords))


# solution = perform_tabu_search_better(50, 100, 3000, city_coords)
# print(solution)
# print(calc_route_length(solution, city_coords))
# draw_route(solution, city_coords)

# results = {}
# for i in range(0, 10):
#     (l, c, t) = tune_parameters_sa(0.05, 0.0, 1, 0, 1, 100, 3000, city_coords)
#     results[l] = t
# od = collections.OrderedDict(sorted(results.items()))
# print(od)

# (solutions, lengths, average_length, standard_dev) = do_sa_runs(50, 0.05, 3000, 30, city_coords)
# print(average_length)
# print(standard_dev)

# solution = perform_genetic_algorithm(3000, 0.5, 2, 0.5, 3000, city_coords)
# # solution = read_tour_file("att48.opt.tour")
# print(solution)
# print(calc_route_length(solution, city_coords))
# draw_route(solution, city_coords)

# print(tune_parameters_ga(500, 500, 2, 0.3, 0.3, 3, 0.3, 0.3, 3, 0.3, 0.3, 3, 3000, city_coords))



# print(tune_parameters_sa(0.05, 0.0, 1, 0, 1, 100, 3000, city_coords))

# solution = perform_simulated_annealing(800, 12, 3000, city_coords)
# print(solution)
# print(len(solution))
# print(get_energy(solution, city_coords))
# draw_route(solution, city_coords)

# draw_route(read_tour_file("att48.opt.tour"), city_coords)



# assert(len(solution) == len(set(solution)))

# solution = perform_tabu_sids_and_lengthscoords)
# print(solution)
# print(get_energy(solution, city_coords))

# solution = perform_genetic_algorithm(10, 10, 0.5, 0.5, city_coords)
# print(solution)
# print(get_energy(solution, city_coords))







# route = read_tour_file("att48.opt.tour")
# print(calc_route_length(route, city_coords))
