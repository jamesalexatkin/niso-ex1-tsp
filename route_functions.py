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
