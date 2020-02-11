from route_functions import *


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

def select_parents(pop, pop_size, elitism_size):
    """Select promising parent solutions from a population. Implemented as simple truncation (or elitism) selection.
    
    Args:
        pop (list): List of routes.
        pop_size (int): Number of routes within the population.
        elitism_size (float): Number in range [0,1] representing the top proportion of solutions to take forward.
    
    Returns:
        list: List of routes.
    """
    # Simple truncation/elitism selection
    group_size = math.floor(pop_size * elitism_size)

    return pop[0:group_size]

def breed_route(parent1, parent2):
    """Produce a child route from two parents.
    
    Args:
        parent1 (list): List of point numbers composing a route.
        parent2 (list): List of point numbers composing a route.
    
    Returns:
        list: List of point numbers composing a route.
    """
    child = []

    bound1 = random.randint(0, len(parent1))
    bound2 = random.randint(0, len(parent2))

    lower_bound = min(bound1, bound2)
    upper_bound = max(bound1, bound2)

    for i in range(0, len(parent1)):
        if i in range(lower_bound, upper_bound):
            child.append(parent1[i])
        else:
            child.append(parent2[i])
            
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

def mutate_route(route, mutation_prob):
    """Mutate a route by randomly swapping cities around.
    
    Args:
        route (list): List of point numbers composing a route.
        mutation_prob (float): Probability with which to perform a mutation.
    
    Returns:
        list: List of point numbers composing a route.
    """
    mutant = route.copy()
    for i in range(0, len(route) - 1):
        if(random.random() < mutation_prob):
            j = random.randint(0, len(route) - 1)

            city1 = route[i]
            city2 = route[j]

            mutant[i] = city2
            mutant[j] = city1
    return mutant

def mutate_population(child_routes, mutation_prob):
    """Mutate a population of routes. 
    
    Args:
        child_routes (list): List of routes to mutate.
        mutation_prob (float): Probability with which to perform a mutation.
    
    Returns:
        list: List of routes.
    """
    mutated_pop = []

    for route in child_routes:
        mutant = mutate_route(route, mutation_prob)
        mutated_pop.append(mutant)

    return mutated_pop

def produce_next_gen(pop, pop_size, elitism_size, mutation_prob, city_coords):
    """Produce a new generation of routes. Goes through the stages of fitness evaluation, selection, reproduction and mutation.
    
    Args:
        pop (list): List of routes forming the previous generation.
        pop_size (int): Number of routes within the population.
        elitism_size (float): Number in range [0,1] representing the top proportion of solutions to take forward.
        mutation_prob (float): Probability with which to perform a mutation.
        city_coords (dict): Dictionary containing the number of the node as key, and an (x, y) coordinate as value e.g. {1 : (34, 56)}.
    
    Returns:
        list: List of routes representing the new generation.
    """
    # Evaluation of fitness
    pop = rank_routes(pop, city_coords)

    # Selection
    selected_routes = select_parents(pop, pop_size, elitism_size)

    # Reproduction
    child_routes = breed_population(selected_routes, pop_size)

    # Variation
    next_gen = mutate_population(child_routes, mutation_prob)

    return next_gen

def perform_genetic_algorithm(pop_size, max_iteration, elitism_size, mutation_prob, city_coords):
    """Perform a genetic algorithm (GA) to solve the Travelling Salesman Problem for a given group of cities.
    
    Args:
        pop_size (int): Number of routes within the population.
        max_iteration (int): Maximum number of iterations of the algorithm to perform.
        elitism_size (float): Number in range [0,1] representing the top proportion of solutions to take forward.
        mutation_prob (float): Probability with which to perform a mutation.
        city_coords (dict): Dictionary containing the number of the node as key, and an (x, y) coordinate as value e.g. {1 : (34, 56)}.
    
    Returns:
        list: List of point numbers composing the best route found.
    """
    pop = generate_rand_population(pop_size, city_coords)

    iteration = 0

    while iteration < max_iteration:        
        pop = produce_next_gen(pop, pop_size, elitism_size, mutation_prob, city_coords)
        iteration = iteration + 1

    sol_best = rank_routes(pop, city_coords)[0]

    return sol_best