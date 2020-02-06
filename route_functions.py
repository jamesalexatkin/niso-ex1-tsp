import math
import random


def read_tsp_file(filename):
    """Return a dictionary of integer coordinates read from a .tsp file.
    
    Arguments:
        filename (string): Name of the file to read
    
    Returns:
        dict: Dictionary containing the number of the node as key, and an (x, y) coordinate as value e.g. {1 : (34, 56)}
    """
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
    """Return a route as read from a .tour file.
    
    Arguments:
        filename(string): Name of the file to read
    
    Returns:
        list: List of point numbers composing a route
    """
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
    """Calculate the pseudo-Euclidean distance between two (x, y) points.
    
    Arguments:
        i (int, int): Tuple representing x and y as integers
        j  (int, int): Tuple representing x and y as integers
    
    Returns:
        int: Integer representing the calculated distance
    """
    xdiff = i[0] - j[0]
    ydiff = i[1] - j[1]
    rij = math.sqrt((xdiff * xdiff + ydiff * ydiff) / 10.0)
    # Skip out some of the faffing in the PDF function by just always rounding up
    dij = math.ceil(rij)
    
    return dij

def generate_rand_route(city_coords):
    """Generate a random route between all cities, given a dict.
    
    Args:
        city_coords (dict): Dictionary containing the number of the node as key, and an (x, y) coordinate as value e.g. {1 : (34, 56)}
    
    Returns:
        list: List of point numbers composing a route
    """
    cities = city_coords.keys()
    route = random.sample(cities, len(cities))
    route.append(route[0])
    return route

def calc_route_length(route, city_coords):
    """Calculate the length of a route.
    
    Args:
        route (list): List of point numbers composing a route
        city_coords Dictionary containing the number of the node as key, and an (x, y) coordinate as value e.g. {1 : (34, 56)}
    
    Returns:
        int: Length of route
    """
    length = 0
    i = 0
    while i != len(route)-1:
        length = length + calc_pseudo_euclid_dist(city_coords[route[i]], city_coords[route[i+1]])
        i = i + 1
    length = length + calc_pseudo_euclid_dist(city_coords[route[i]], city_coords[route[0]])
    
    return length