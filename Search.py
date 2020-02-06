import math
import random


def read_tsp_file(filename):
    coords = {}
    f = open(filename, "r")
    for line in f:
        if line == "NODE_COORD_SECTION\n":
            line = f.readline()
            while line != "EOF\n":
                tokens = line.split(" ")
                coords[int(tokens[0])] = (int(tokens[1]), int(tokens[2].rstrip()))
                line = f.readline()
    f.close()
    return coords

def read_tour_file(filename):
    route = []
    f = open(filename, "r")
    for line in f:
        if line == "TOUR_SECTION\n":
            line = f.readline()
            while line != "-1\n":
                tokens = line.split(" ")
                route.append(int(line.rstrip()))
                line = f.readline()
    f.close()
    return route

def calc_pseudo_euclid_dist(i, j):
    xdiff = i[0] - j[0]
    ydiff = i[1] - j[1]
    rij = math.sqrt((xdiff * xdiff + ydiff * ydiff) / 10.0)
    # Skip out some of the faffing in the PDF function by just always rounding up
    dij = math.ceil(rij)
    
    return dij

def generate_rand_route(city_coords):
    cities = city_coords.keys()
    route = random.sample(cities, len(cities))
    route.append(route[0])
    return route

def calc_route_length(route, city_coords):
    length = 0
    i = 0
    while i != len(route)-1:
        length = length + calc_pseudo_euclid_dist(city_coords[route[i]], city_coords[route[i+1]])
        i = i + 1
    length = length + calc_pseudo_euclid_dist(city_coords[route[i]], city_coords[route[0]])
    
    return length

def get_energy(solution, city_coords):
    return calc_route_length(solution, city_coords)
    
def get_rand_neighbour(solution):
    max_pos = len(solution) - 1
    rand_pos_1 = random.randint(2, max_pos)
    rand_pos_2 = random.randint(0, max_pos - rand_pos_1)

    neighbour = solution.copy()
    neighbour[rand_pos_1 : (rand_pos_1 + rand_pos_2)] = reversed(neighbour[rand_pos_1 : (rand_pos_1 + rand_pos_2)])

    return neighbour

def calc_acceptance_prob(energy_cur, energy_new, temp):
    if energy_new < energy_cur:
        return 1
    else:
        return math.exp((energy_cur - energy_new) / temp)

def calc_new_temperature(temp, cooling_rate):
    return temp = temp * cooling_rate

def perform_simulated_annealing(temp_initial, cooling_rate, max_iteration, city_coords):
    sol_initial = generate_rand_route(city_coords)
    sol_cur = sol_initial
    energy_cur = get_energy(sol_cur, city_coords)
    sol_best = sol_cur
    energy_best = energy_cur

    temp = temp_initial
    iteration = 0

    while temp >= 0 and iteration < max_iteration:
        sol_new = get_rand_neighbour(sol_cur)
        energy_new = get_energy(sol_new, city_coords)

        if calc_acceptance_prob(energy_cur, energy_new, temp) > random.uniform(0, 1):
            sol_cur = sol_new
            energy_cur = energy_new
        if energy_new < energy_best:
            sol_best = sol_new
            energy_best = energy_new

        temp = calc_new_temperature(temp, cooling_rate)
        iteration = iteration + 1

    return sol_best



# Program start

city_coords = read_tsp_file("att48.tsp")
solution = perform_simulated_annealing(100, 0.7, 50, city_coords)

print(solution)
print(get_energy(solution, city_coords))

# route = read_tour_file("att48.opt.tour")
# print(calc_route_length(route, city_coords))