import collections
import math
import random
import statistics
from tqdm import tqdm

import matplotlib.pyplot as plt


########## GENERIC ROUTE FUNCTIONS ##########

def read_tsp_file(filename):
    """Return a dictionary of integer coordinates read from a .tsp file.
    
    Arguments:
        filename (string): Name of the file to read.
    
    Returns:
        dict: Dictionary containing the number of the node as key, and an (x, y) coordinate as value e.g. {1 : (34, 56)}.
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
        filename(string): Name of the file to read.
    
    Returns:
        list: List of point numbers composing a route.
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
        i (int, int): Tuple representing x and y as integers.
        j  (int, int): Tuple representing x and y as integers.
    
    Returns:
        int: Integer representing the calculated distance.
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
        city_coords (dict): Dictionary containing the number of the node as key, and an (x, y) coordinate as value e.g. {1 : (34, 56)}.
    
    Returns:
        list: List of point numbers composing a route.
    """
    route = [*city_coords]
    random.shuffle(route)
    return route

def calc_route_length(route, city_coords):
    """Calculate the length of a route.
    
    Args:
        route (list): List of point numbers composing a route.
        city_coords Dictionary containing the number of the node as key, and an (x, y) coordinate as value e.g. {1 : (34, 56)}.
    
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

def draw_route(route, city_coords):
    """Visualise a route. Points represent cities and edges represent a journey between them.
    
    Args:
        route (list): List of point numbers composing a route.
        city_coords Dictionary containing the number of the node as key, and an (x, y) coordinate as value e.g. {1 : (34, 56)}.
    """
    for i in range(0, len(route)):
        city_num1 = route[i]
        (x1, y1) = city_coords[city_num1]

        if i == 0:
            plt.text(x1+0.3, y1+0.3, str(city_num1) + " Start", fontsize=9)
        elif i == len(route)-1:
            plt.text(x1+0.3, y1+0.3, str(city_num1) + " End", fontsize=9)
        else:
            plt.text(x1+0.3, y1+0.3, city_num1, fontsize=9)

        if i == len(route) - 1:
            city_num2 = route[0]        
            (x2, y2) = city_coords[city_num2]
        else:
            city_num2 = route[i+1]        
            (x2, y2) = city_coords[city_num2]

        plt.plot([x1,x2], [y1,y2], 'ro-')

    plt.show()


########## SIMULATED ANNEALING FUNCTIONS ##########
def get_energy(solution, city_coords):
    """Get the energy of a solution. In the Travelling Salesman Problem, this is the length of the route given by the solution.
    
    Args:
        solution (list): List of point numbers composing a route
        city_coords (dict): Dictionary containing the number of the node as key, and an (x, y) coordinate as value e.g. {1 : (34, 56)}
    
    Returns:
        int: Length of route
    """
    return calc_route_length(solution, city_coords)
    
def get_rand_neighbour(solution):
    """Get a random neighbouring solution given a particular solution. This is found by picking two random points and swapping their positions in the route.
    
    Args:
        solution (list): List of point numbers composing a route
    
    Returns:
        list: List of point numbers composing a route
    """
    max_pos = len(solution) - 1
    rand_pos_1 = random.randint(0, max_pos)
    rand_pos_2 = rand_pos_1
    while rand_pos_2 == rand_pos_1:
        rand_pos_2 = random.randint(0, max_pos)

    city_1 = solution[rand_pos_1]
    city_2 = solution[rand_pos_2]

    neighbour = solution.copy()
    neighbour[rand_pos_1] = city_2
    neighbour[rand_pos_2] = city_1

    return neighbour

def get_rand_neighbour_2opt(solution):
    """Get a random neighbouring solution given a particular solution. This is found by performing a 2-Opt swap between two randoms cities.
    
    Args:
        solution (list): List of point numbers composing a route
    
    Returns:
        list: List of point numbers composing a route
    """
    # Get two randoms
    max_pos = len(solution) - 1
    rand_pos_1 = random.randint(0, max_pos)
    rand_pos_2 = rand_pos_1
    # Make sure second random is distinct and not same as first
    while rand_pos_2 == rand_pos_1:
        rand_pos_2 = random.randint(0, max_pos)

    i = min(rand_pos_1, rand_pos_2)
    j = max(rand_pos_1, rand_pos_2)

    neighbour = []

    # Python slice operator is exclusive on the upper bound so the calculations are slightly different
    # Items 1 - (i-1) get added in order
    for city in solution[0 : i]:
        neighbour.append(city)
    # Items i - j get added in reversed order
    for city in reversed(solution[i : (j+1)]):
        neighbour.append(city)
    # Items j+1 - end get added in order
    for city in solution[(j+1) :]:
        neighbour.append(city)

    return neighbour

def calc_acceptance_prob(energy_cur, energy_new, temp):
    """Calculate the probability needed to accept a new solution as the best solution.
    
    Args:
        energy_cur (int): Energy (distance) of the current solution (route)
        energy_new (int): Energy (distance) of the new solution (route)
        temp (float): Temperature of the annealing process
    
    Returns:
        float: The probabiliy
    """
    if energy_new < energy_cur:
        return 1.0
    else:
        return math.exp((energy_cur - energy_new) / temp)

def linear_cooling(temp_initial, cooling_rate, iteration):
    """Get new temperature via linear multiplicative cooling.
    
    Args:
        temp_initial (float): Initial temperature of the annealing process
        cooling_rate (float): Factor by which to decrease the temperature
        iteration (int): Iteration the main SA algorithm is currently on
    
    Returns:
        float: New temperature of the annealing process
    """
    return temp_initial / (1 + (cooling_rate * iteration))

def perform_simulated_annealing(temp_initial, cooling_rate, max_iteration, city_coords):
    """Perform simulated annealing (SA) to solve the Travelling Salesman Problem for a given group of cities.
    
    Args:
        temp_initial (float): Initial temperature of the annealing process
        cooling_rate (float): Factor by which to decrease the temperature
        max_iteration (int): Maximum number of interations of the algorithm to perform
        city_coords (dict): Dictionary containing the number of the node as key, and an (x, y) coordinate as value e.g. {1 : (34, 56)}
    
    Returns:
        list: List of point numbers composing the best route found
    """
    sol_initial = generate_rand_route(city_coords)
    sol_cur = sol_initial
    energy_cur = get_energy(sol_cur, city_coords)
    sol_best = sol_cur
    energy_best = energy_cur

    temp = temp_initial
    iteration = 0

    while temp > 0 and iteration < max_iteration:
        sol_new = get_rand_neighbour_2opt(sol_cur)
        energy_new = get_energy(sol_new, city_coords)

        if calc_acceptance_prob(energy_cur, energy_new, temp) > random.uniform(0, 1):
            sol_cur = sol_new
            energy_cur = energy_new
        if energy_new < energy_best:
            sol_best = sol_new
            energy_best = energy_new

        temp = linear_cooling(temp_initial, cooling_rate, iteration)
        iteration = iteration + 1

    return sol_best

def tune_parameters_sa(min_temp, temp_step, temp_iter, min_cooling_rate, cooling_step, cooling_iter, max_iteration, city_coords):
    """Tune the parameters of the SA algorithm across a certain ranges
    
    Args:
        min_temp (float): Minimum bound for the temperature
        temp_step (int): Interval across which to measure the temperature
        temp_iter (int): Number of values to try for the temperature
        min_cooling_rate (float): Minimum bound for the cooling rate
        cooling_step (int): Interval across which to measure the cooling rate
        cooling_iter (int): Number of values to try for the cooling rate
        max_iteration (int): Maximum number of interations of the algorithm to perform
        city_coords (dict): Dictionary containing the number of the node as key, and an (x, y) coordinate as value e.g. {1 : (34, 56)}
    
    Returns:
        list: List of tuples representing (route_length, temperature, cooling_rate) for a run of SA
    """
    cooling_rate = min_cooling_rate
    temperature = min_temp

    i = 0

    results = []

    while i < temp_iter:
        j = 0
        cooling_rate = min_cooling_rate
        while j < cooling_iter:
            solution = perform_simulated_annealing(temperature, cooling_rate, max_iteration, city_coords)
            length = calc_route_length(solution, city_coords)
            results.append((length, temperature, cooling_rate))

            print("Completed run " + str(i) + "/" + str(temp_iter-1) + ", " + str(j) + "/" + str(cooling_iter-1), end="\r")
            cooling_rate = cooling_rate + cooling_step
            j = j + 1
        temperature = temperature + temp_step
        i = i + 1

    print("\n")
    return results

def find_best_parameters_sa(results):
    """Find the best parameters, given a set of results.
    
    Args:
        results (list): List of tuples representing (route_length, temperature, cooling_rate) for a run of SA
    
    Returns:
        (best_length, best_temp, best_cooling_rate): Tuple containing the best route length from the results, along with the parameters that generated it
    """
    best_temp = 0
    best_cooling_rate = 0
    best_length = float("inf")

    for (length, temp, cooling) in results:
        if length < best_length:
            best_length = length
            best_temp = temp
            best_cooling_rate = cooling

    return (best_length, best_temp, best_cooling_rate)

def do_sa_runs(temperature, cooling_rate, max_iteration, max_runs, city_coords):
    """Perform a number of runs of the whole SA algorithm.
    
    Args:
        temp_initial (float): Initial temperature of the annealing process
        cooling_rate (float): Factor by which to decrease the temperature
        max_iteration (int): Maximum number of interations of the algorithm to perform
        max_runs (int): Maximum number of runs to perform
        city_coords (dict): Dictionary containing the number of the node as key, and an (x, y) coordinate as value e.g. {1 : (34, 56)}
    
    Returns:
        (solutions, lengths, best_solution, best_length, average_length, standard_dev): Tuple containing list of routes generated, lengths of the routes, the best route, length of best route, average route length across routes, standard deviation of route lengths
    """

    solutions = []
    lengths = []
    cumulative_lengths = 0

    best_length = float("inf")
    best_solution = []

    for i in range(0, max_runs):
        solution = perform_simulated_annealing(temperature, cooling_rate, max_iteration, city_coords)
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


########## GENETIC ALGORITHM FUNCTIONS ##########

def generate_rand_population(pop_size, city_coords):
    """Generate a random population of a given size.
    
    Args:
        pop_size (int): Number of individuals within the population.
        city_coords (dict): Dictionary containing the number of the node as key, and an (x, y) coordinate as value e.g. {1 : (34, 56)}.
    
    Returns:
        list: List of routes.
    """
    pop = []

    for i in range(0, pop_size):
        pop.append(generate_rand_route(city_coords))

    return pop

def rank_routes(pop, city_coords):
    """Rank routes in ascending order according to their length.
    
    Args:
        pop (list): List of routes.
        city_coords (dict): Dictionary containing the number of the node as key, and an (x, y) coordinate as value e.g. {1 : (34, 56)}.
    
    Returns:
        list: List of routes.
    """
    ids_and_lengths = {}

    for i in range(0, len(pop)):
        ids_and_lengths[i] = calc_route_length(pop[i], city_coords)
    ranked_ids = {k: v for k, v in sorted(ids_and_lengths.items(), key=lambda item: item[1])}

    ranked_routes = []

    for id in ranked_ids.keys():
        ranked_routes.append(pop[id])

    return ranked_routes

def select_parents(pop, pop_size, elite_proportion, norm_factor):
    """Select promising parent solutions from a population. Implemented as simple truncation (also known as elitism) selection.
    
    Args:
        pop (list): List of routes.
        pop_size (int): Number of routes within the population.
        elitism_size (float): Number in range [0,1] representing the top proportion of solutions to take forward.
    
    Returns:
        list: List of routes.
    """
    selected_routes = []
    
    # Simple truncation/elitism selection
    elite_group_size = math.floor(pop_size * elite_proportion)

    for i in range(0, elite_group_size):
        selected_routes.append(pop[i])

    for i in range(elite_group_size, pop_size):
        random_prob = random.random()
        # Exponential ranking function
        selection_prob = (1 - math.exp(0-i)) / norm_factor

        if random_prob > selection_prob:
            selected_routes.append(pop[i])

    return selected_routes

def breed_route(parent1, parent2):
    """Produce a child route from two parents. Uses 2-point crossover.
    
    Args:
        parent1 (list): List of point numbers composing a route.
        parent2 (list): List of point numbers composing a route.
    
    Returns:
        list: List of point numbers composing a route.
    """
    child = []

    num1 = random.randint(0, len(parent1))
    num2 = random.randint(0, len(parent2))

    # Use smaller number as lower bound and larger number as upper bound
    start = min(num1, num2)
    end = max(num1, num2)

    # Find the slice of the parent route we are using
    slice1 = parent1[start : end]
    # Maintain a pointer in the second route to track what we've included
    ptr2 = 0

    for ptr1 in range(0, len(parent1)):
        if ptr1 in range(start, end):
            child.append(parent1[ptr1])
        else:
            # Only do this if second pointer within range
            if ptr2 < len(parent1):
                gene2 = parent2[ptr2]
                # Wind pointer forward to next available city in route 2
                while gene2 in slice1:
                    ptr2 = ptr2 + 1
                    if ptr2 >= len(parent1):
                        break
                    else:
                        gene2 = parent2[ptr2]
                child.append(gene2)
                ptr2 = ptr2 + 1

    return child

def breed_population(selected_routes, pop_size):
    """Breed a new population of a given size from some parent routes.
    
    Args:
        selected_routes (list): List of routes representing parents to breed.
        pop_size (int): Number of routes within the population.
    
    Returns:
        list: List of routes representing a new population.
    """
    children = []

    while(len(children) < pop_size):
        for i in range(0, len(selected_routes) - 1):
            # Get another parent j
            j = i
            # Make sure we have distinct parents
            while j == i:
                j = random.randint(0, len(selected_routes) - 1)

            child = breed_route(selected_routes[i], selected_routes[j])
            children.append(child)

    return children

def produce_next_gen(pop, pop_size, elitism_size, norm_factor, city_coords):
    """Produce a new generation of routes. Goes through the stages of fitness evaluation, selection, reproduction and mutation.
    
    Args:
        pop (list): List of routes forming the previous generation.
        pop_size (int): Number of routes within the population.
        elitism_size (float): Number in range [0,1] representing the top proportion of solutions to take forward.
        norm_factor (float): Normalising factor used in selection.
        city_coords (dict): Dictionary containing the number of the node as key, and an (x, y) coordinate as value e.g. {1 : (34, 56)}.
    
    Returns:
        list: List of routes representing the new generation.
    """
    # Evaluation of fitness
    pop = rank_routes(pop, city_coords)

    # Selection
    selected_routes = select_parents(pop, pop_size, elitism_size, norm_factor)

    # Reproduction
    child_routes = breed_population(selected_routes, pop_size)

    return child_routes

def perform_genetic_algorithm(pop_size, elitism_size, norm_factor, max_iteration, city_coords):
    """Perform a genetic algorithm (GA) to solve the Travelling Salesman Problem for a given group of cities.
    
    Args:
        pop_size (int): Number of routes within the population.
        elitism_size (float): Number in range [0,1] representing the top proportion of solutions to take forward.
        norm_factor (float): Normalising factor used in selection.
        max_iteration (int): Maximum number of iterations of the algorithm to perform.
        city_coords (dict): Dictionary containing the number of the node as key, and an (x, y) coordinate as value e.g. {1 : (34, 56)}.
    
    Returns:
        list: List of point numbers composing the best route found.
    """
    pop = generate_rand_population(pop_size, city_coords)
    best_length = float("inf")

    for i in tqdm(range(0, max_iteration)):
        pop = produce_next_gen(pop, pop_size, elitism_size, norm_factor, city_coords)

        current_best_length = calc_route_length(pop[0], city_coords)

        if current_best_length != best_length:
            best_length = current_best_length

        if i % 10 == 0:
            tqdm.write(" Best length : " + str(current_best_length))

    sol_best = rank_routes(pop, city_coords)[0]

    return sol_best

def tune_parameters_ga(min_pop_size, pop_step, pop_iter, min_elite, elite_step, elite_iter, min_norm, norm_step, norm_iter, max_iteration, city_coords):
    """Tune the parameters of the GA algorithm across given ranges
    
    Args:
        min_pop_size (int): Minimum bound for the population size
        pop_step (int): Interval across which to measure the population size
        pop_iter (int): Number of values to try for the population size
        min_elite (float): Minimum bound for the elitism proportion
        elite_step (int): Interval across which to measure the elitism proportion
        elite_iter (int): Number of values to try for the elitism proportion
        min_norm (float): Minimum bound for the normalising factor
        norm_step (int): Interval across which to measure the normalising factor
        norm_iter (int): Number of values to try for the normalising factor
        max_iteration (int): Maximum number of interations of the algorithm to perform
        city_coords (dict): Dictionary containing the number of the node as key, and an (x, y) coordinate as value e.g. {1 : (34, 56)}
    
    Returns:
        list: List of tuples representing (route_length, temperature, cooling_rate) for a run of GA
    """
    pop_size = min_pop_size
    elitism_size = min_elite
    norm_factor = min_norm

    results = []

    i = 0
    for i in tqdm(range(0, pop_iter)):
        j = 0
        elitism_size = min_elite
        for j in tqdm(range(0, elite_iter)):
            k = 0
            norm_factor = min_norm
            for k in tqdm(range(0, norm_iter)):

                solution = perform_genetic_algorithm(pop_size, elitism_size, norm_factor, max_iteration, city_coords)
                length = calc_route_length(solution, city_coords)
                results.append((length, pop_size, elitism_size, norm_factor))
            
                norm_factor = norm_factor + norm_step
            elitism_size = elitism_size + elite_step
        pop_size = pop_size + pop_step
        
    print("\n")
    return results

def find_best_parameters_ga(results):
    """Find the best parameters, given a set of results.
    
    Args:
        results (list): List of tuples representing (route_length, pop_size, elitism_size, norm_factor) for a run of GA
    
    Returns:
        (best_length, best_pop_size, best_elitism_size, best_norm_factor): Tuple containing the best route length from the results, along with the parameters that generated it
    """
    best_pop_size = 0
    best_elitism_size = 0
    best_norm_factor = 0
    best_length = float("inf")

    for (length, pop_size, elitism_size, norm_factor) in results:
        if length < best_length:
            best_length = length
            best_pop_size = pop_size
            best_elitism_size = elitism_size
            best_norm_factor = norm_factor

    return (best_length, best_pop_size, best_elitism_size, best_norm_factor)

def do_ga_runs(pop_size, elitism_size, norm_factor, max_iteration, max_runs, city_coords):
    """Perform a number of runs of the whole GA algorithm.
    
    Args:
        pop_size (int): Population size
        elitism_size (float): Proportion of the population to use in elitism
        norm_factor (float): Normalising factor for the selection process
        max_iteration (int): Maximum number of interations of the algorithm to perform
        max_runs (int): Maximum number of runs to perform
        city_coords (dict): Dictionary containing the number of the node as key, and an (x, y) coordinate as value e.g. {1 : (34, 56)}
    
    Returns:
        (solutions, lengths, best_solution, best_length, average_length, standard_dev): Tuple containing list of routes generated, lengths of the routes, the best route, length of best route, average route length across routes, standard deviation of route lengths    
    """

    solutions = []
    lengths = []
    # Accumulate lengths so we can average at the end
    cumulative_lengths = 0

    best_length = float("inf")
    best_solution = []

    for i in range(0, max_runs):
        solution = perform_genetic_algorithm(pop_size, elitism_size, norm_factor, max_iteration, city_coords)
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


########## TABU SEARCH FUNCTIONS ##########

def get_neighbours_2opt(solution, neighbourhood_size):
    """Get a given number of neighbours of a solution via 2-opt.
    
    Args:
        solution ([type]): [description]
        neighbourhood_size (integer): Number of neighbours to find
    
    Returns:
        (neighbours, cities_swapped): Tuple containing list of neighbour routes, and list of pairs of cities which were swapped to make the nieghbour
    """
    max_pos = len(solution) - 1
    neighbours = []
    cities_swapped = []

    for n in range(0, neighbourhood_size):
        max_pos = len(solution) - 1
        rand_pos_1 = random.randint(0, max_pos)
        rand_pos_2 = rand_pos_1
        # Make sure second random is distinct and not same as first
        while rand_pos_2 == rand_pos_1:
            rand_pos_2 = random.randint(0, max_pos)

        i = min(rand_pos_1, rand_pos_2)
        j = max(rand_pos_1, rand_pos_2)

        new_neighbour = []

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

        # Only add neighbour if we don't have it already
        if new_neighbour not in neighbours:
            neighbours.append(new_neighbour)
            cities_swapped.append((i, j))

    return (neighbours, cities_swapped)


def perform_tabu_search(max_tabu, neighbourhood_size, max_iteration, city_coords):
    """Perform tabu search (TS)) to solve the Travelling Salesman Problem for a given group of cities.
    
    Args:
        max_tabu (int): Maximum number of elements to store in the tabu list.
        neighbourhood_size (int): Maximum number of neighbours to find when considering neighbourhood solutions.
        max_iteration (int): Maximum number of interations of the algorithm to perform
        city_coords (dict): Dictionary containing the number of the node as key, and an (x, y) coordinate as value e.g. {1 : (34, 56)}
    
    Returns:
        list: List of point numbers composing the best route found
    """
    # Randomly generate a route to start
    sol_initial = generate_rand_route(city_coords)
    sol_best = sol_initial
    length_best = float("inf")

    tabu_list = []

    candidate_best = sol_initial

    for i in tqdm(range(0, max_iteration), miniters=10):
        # Find neighbour solutions for the best candidate
        (neighbours, cities_swapped) = get_neighbours_2opt(candidate_best, neighbourhood_size)
        # Start by considering first neighbour
        candidate_best = neighbours[0]
        candidate_best_length = calc_route_length(candidate_best, city_coords)
        candidate_best_index = 0
        for i in range(0, len(neighbours)):
            candidate = neighbours[i]
            candidate_length = calc_route_length(candidate, city_coords)
            candidate_city1 = cities_swapped[i][0]
            candidate_city2 = cities_swapped[i][1]

            # Only add route if it doesn't swap tabu cities
            if (candidate_city1 not in tabu_list) and (candidate_city2 not in tabu_list) and (candidate_length < candidate_best_length):
                candidate_best = candidate
                candidate_best_length = candidate_length
                candidate_best_index = i

        # If the best candidate is better than the overall best route, that's now the best route
        if candidate_best_length < length_best:
            sol_best = candidate_best
            length_best = candidate_best_length
        
        # Best candidate cities now tabu
        if candidate_city1 in tabu_list:
            # Remove and then append to push city to back of list, given that it has now been used more recently
            tabu_list.remove(candidate_city1)
            tabu_list.append(candidate_city1)
        else:
            tabu_list.append(candidate_city1)
        if candidate_city2 in tabu_list:
            tabu_list.remove(candidate_city2)
            tabu_list.append(candidate_city2)
        else:
            tabu_list.append(candidate_city2)

        # Remove oldest tabu cities if we have too many tabus
        while len(tabu_list) > max_tabu:
            tabu_list.pop(0)

        # Print best length every 10 iterations
        if i % 10 == 0:
            tqdm.write(" Best length : " + str(length_best))

    return sol_best


def tune_parameters_ts(min_tabu_size, tabu_size_step, tabu_size_iter, min_neighbourhood_size, neighbourhood_size_step, neighbourhood_size_iter, max_iteration, city_coords):
    """Tune the parameters of the TS algorithm across given ranges
    
    Args:
        min_tabu_size (int): Minimum bound for the tabu list length
        tabu_size_step (int): Interval across which to measure the tabu list length
        tabu_size_iter (int): Number of values to try for the tabu list length
        min_neighbourhood_size (int): Minimum bound for the neighbourhood size
        neighbourhood_size_step (int): Interval across which to measure the neighbourhood size
        neighbourhood_size_iter (int): Number of values to try for the neighbourhood size
        max_iteration (int): Maximum number of interations of the algorithm to perform
        city_coords (dict): Dictionary containing the number of the node as key, and an (x, y) coordinate as value e.g. {1 : (34, 56)}
    
    Returns:
        list: List of tuples representing (route_length, temperature, cooling_rate) for each run of TS
    """
    tabu_size = min_tabu_size

    results = []

    # Loop over tabu sizes
    for i in tqdm(range(0, tabu_size_iter)):
        neighbourhood_size = min_neighbourhood_size
        # Look over neighbourhood sizes
        for j in tqdm(range(0, neighbourhood_size_iter)):
            solution = perform_tabu_search(tabu_size, neighbourhood_size, max_iteration, city_coords)
            length = calc_route_length(solution, city_coords)
            results.append((length, tabu_size, neighbourhood_size))
            neighbourhood_size = neighbourhood_size + neighbourhood_size_step
            print("Completed run " + str(i+1) + "/" + str(tabu_size_iter) + ", " + str(j+1) + "/" + str(neighbourhood_size_iter))

        tabu_size = tabu_size + tabu_size_step

    print("\n")
    return results

def find_best_parameters_ts(results):
    """Find the best parameters, given a set of results.
    
    Args:
        results (list): List of tuples representing (route_length, tabu_size, neighbourhood_size) for a run of TS
    
    Returns:
        (best_length, best_tabu, best_neighbourhood_size): Tuple containing the best route length from the results, along with the parameters that generated it
    """
    best_tabu_size = 0
    best_neighbourhood_size = 0
    best_length = float("inf")

    for (length, tabu_size, neighbourhood_size) in results:
        if length < best_length:
            best_length = length
            best_tabu = tabu_size

    return (best_length, best_tabu, best_neighbourhood_size)

def do_ts_runs(tabu_size, neighbourhood_size, max_iteration, max_runs, city_coords):
    """Perform a number of runs of the whole TS algorithm.
    
    Args:
        tabu_size (int): Size of the tabu list to use
        neighbourhood_size (int): Size of the neighbourhood to consider for neighbour solutions
        max_iteration (int): Maximum number of interations of the algorithm to perform
        max_runs (int): Maximum number of runs to perform
        city_coords (dict): Dictionary containing the number of the node as key, and an (x, y) coordinate as value e.g. {1 : (34, 56)}
    
    Returns:
        (solutions, lengths, best_solution, best_length, average_length, standard_dev): Tuple containing list of routes generated, lengths of the routes, the best route, length of best route, average route length across routes, standard deviation of route lengths    
    """

    solutions = []
    lengths = []
    cumulative_lengths = 0

    best_length = float("inf")
    best_solution = []

    for i in tqdm(range(0, max_runs)):
        solution = perform_tabu_search(tabu_size, neighbourhood_size, max_iteration, city_coords)
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


########### MAIN METHOD ###########
### This is what will run when you do "python submission.py"
### I've commented everything out so far, but you can uncomment the bits you wish to run
### For each of the three algorithms, there is code to tune and find parameters, followed by code to run the algorithm
### For running the algorithms you must specify the parameters you wish to use
### You can also try playing around with the number of iterations etc.
### I've included the att48.tsp and att48.opt.tour files in the directory with this code. They must be in the same directory for the code to run properly

# Reading in the coordinate file to an object
# You need to do this before running any of the algorithms so I've left it uncommented
city_coords = read_tsp_file("att48.tsp")

### DRAWING THE OPTIMAL ROUTE
# route = read_tour_file("att48.opt.tour")
# draw_route(route, city_coords)

### SIMULATED ANNEALING
# Finding parameters
# parameter_results = tune_parameters_sa(100, 100, 6, 2, 2, 5, 3000, city_coords)
# print("Parameter results: ")
# print(parameter_results)
# (best_length, best_temperature, best_cooling_rate) = find_best_parameters_sa(parameter_results)
# print("Best length, best temperature, best cooling size: " + str(best_length) + ", " + str(best_temperature) + ", " + str(best_cooling_rate))

# # Running SA
# (solutions, lengths, best_solution, best_length, average_length, standard_dev) = do_sa_runs(500, 8, 3000, 30, city_coords)
# print("Average length: " + str(average_length))
# print("Standard deviation: " + str(standard_dev))
# print(lengths)
# draw_route(best_solution, city_coords)


### GENETIC ALGORITHM
# # Finding parameters
# parameter_results = tune_parameters_ga(1000, 1000, 5, 0.3, 0.3, 3, 0.5, 0.5, 2, 3000, city_coords)
# print("Parameter results: ")
# print(parameter_results)
# (best_length, best_pop_size, best_elitism_size, best_norm_factor) = find_best_parameters_ga(parameter_results)
# print("Best length, best pop size, best elite size, best norm factor: " + str(best_length) + ", " + str(best_pop_size) + ", " + str(best_elitism_size) + ", " + str(best_norm_factor))

# # Running GA
# (solutions, lengths, best_solution, best_length, average_length, standard_dev) = do_ga_runs(POPULATION SIZE FILL ME IN, ELITISM SIZE FILL ME IN, NORMALISING FACTOR FILL ME IN, 3000, 30, city_coords)
# print("Average length: " + str(average_length))
# print("Standard deviation: " + str(standard_dev))
# print("Best length: " + str(best_length))
# print("Lengths:")
# print(lengths)
# draw_route(best_solution, city_coords)


### TABU SEARCH
# # Finding parameters
# parameter_results = tune_parameters_ts(5, 5, 5, 50, 50, 6, 3000, city_coords)
# print("Parameter results: ")
# print(parameter_results)
# (best_length, best_tabu_size, best_neighbourhood_size) = find_best_parameters_ts(parameter_results)
# print("Best length, best tabu size, best neighbourhood size: " + str(best_length) + ", " + str(best_tabu_size) + ", " + str(best_neighbourhood_size))

# # Running TS
# (solutions, lengths, best_solution, best_length, average_length, standard_dev) = do_ts_runs(TABU SIZE FILL ME IN, NEIGHBOURHOOD SIZE FILL ME IN, 3000, 30, city_coords)
# print("Average length: " + str(average_length))
# print("Standard deviation: " + str(standard_dev))
# print("Best length: " + str(best_length))
# print("Lengths:")
# print(lengths)
# draw_route(best_solution, city_coords)
