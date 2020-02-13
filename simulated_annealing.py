import statistics

from route_functions import *


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

def calc_new_temperature(temp, cooling_rate):
    """Calculate a new temperature according to the annealing schedule.
    
    Args:
        temp (float): Temperature of the annealing process
        cooling_rate (float): Factor by which to decrease the temperature
    
    Returns:
        float: New temperature of the annealing process
    """
    return temp * cooling_rate

def linear_cooling(temp_initial, cooling_rate, iteration):
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

        # temp = calc_new_temperature(temp, cooling_rate)
        temp = linear_cooling(temp_initial, cooling_rate, iteration)
        iteration = iteration + 1

    return sol_best

def tune_parameters_sa(min_temp, temp_step, temp_iter, min_cooling_rate, cooling_step, cooling_iter, max_iteration, city_coords):
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
