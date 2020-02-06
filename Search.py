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
            route.append(route[0])
    f.close()
    return route

def calc_route_length(route, city_coords):
    length = 0
    i = 0
    while i != len(route)-1:
        length = length + calc_pseudo_euclid_dist(city_coords[route[i]], city_coords[route[i+1]])
        i = i + 1
    return length

def calc_pseudo_euclid_dist(i, j):
    xdiff = i[0] - j[0]
    ydiff = i[1] - j[1]
    rij = math.sqrt((xdiff * xdiff + ydiff * ydiff) / 10.0)
    # Skip out some of the faffing in the PDF function by just always rounding up
    dij = math.ceil(rij)
    
    return dij

def simulated_annealing(sol_initial, k_max):
    sol_cur = sol_initial
    energy_cur = get_energy(sol_cur)
    sol_best = sol_cur
    energy_best = energy_cur

    for k in range(0, k_max):
        temp = get_temperature()

        sol_new = get_neighbour(sol_cur)
        energy_new = get_energy(sol_new)

        if calc_acceptance_prob(energy_cur, energy_new, temp) > random.uniform(0, 1):
            sol_cur = sol_new
            energy_cur = energy_new
        if energy_new < energy_best:
            sol_best = sol_new
            energy_best = energy_new

    return sol_best



# Program start

city_coords = read_tsp_file("att48.tsp")

route = read_tour_file("att48.opt.tour")
print(calc_route_length(route, city_coords))