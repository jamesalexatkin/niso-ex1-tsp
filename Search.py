import math
import random

from route_functions import *
from simulated_annealing import *

# Program start
city_coords = read_tsp_file("att48.tsp")
solution = perform_simulated_annealing(100, 0.7, 3000, city_coords)

print(solution)
print(get_energy(solution, city_coords))

# route = read_tour_file("att48.opt.tour")
# print(calc_route_length(route, city_coords))
