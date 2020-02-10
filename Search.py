import math
import random

from genetic_algorithm import *
from route_functions import *
from simulated_annealing import *
from tabu_search import *

# Program start
city_coords = read_tsp_file("att48.tsp")

# solution = perform_simulateids_and_lengths.7, 3000, city_coords)
# print(solution)
# print(get_energy(solution, city_coords))
# assert(len(solution) == len(set(solution)))

# solution = perform_tabu_sids_and_lengthscoords)
# print(solution)
# print(get_energy(solution, city_coords))

solution = perform_genetic_algorithm(10, 10, 0.5, 0.5, city_coords)
print(solution)
print(get_energy(solution, city_coords))

# route = read_tour_file("att48.opt.tour")
# print(calc_route_length(route, city_coords))
