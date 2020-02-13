from tqdm import tqdm

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

def select_parents(pop, pop_size, elite_proportion, norm_factor):
    """Select promising parent solutions from a population. Implemented as simple truncation (or elitism) selection.
    
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

    # print(str(start) + " : " + str(end))
    for ptr1 in range(0, len(parent1)):
        if ptr1 in range(start, end):
            child.append(parent1[ptr1])
        else:
            # Only do this if second pointer within range
            if ptr2 < len(parent1):
                # print(str(ptr1) + ", " + str(ptr2))
                # print(len(parent2))
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

def mutate_route(route, mutation_prob):
    """Mutate a route by randomly swapping cities around.
    
    Args:
        route (list): List of point numbers composing a route.
        mutation_prob (float): Probability with which to perform a mutation.
    
    Returns:
        list: List of point numbers composing a route.
    """
    mutant = route.copy()
    for i in range(0, len(route)):
        if(random.random() < mutation_prob):
            # Random is inclusive so we minus 1 from len
            j = random.randint(0, len(route)-1)

            print(", ".join((str(i), str(j))))


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

def produce_next_gen(pop, pop_size, elitism_size, norm_factor, mutation_prob, city_coords):
    """Produce a new generation of routes. Goes through the stages of fitness evaluation, selection, reproduction and mutation.
    
    Args:
        pop (list): List of routes forming the previous generation.
        pop_size (int): Number of routes within the population.
        elitism_size (float): Number in range [0,1] representing the top proportion of solutions to take forward.
        norm_factor -TODO
        mutation_prob (float): Probability with which to perform a mutation.
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

    # Variation
    # next_gen = mutate_population(child_routes, mutation_prob)
    next_gen = child_routes

    return next_gen

def perform_genetic_algorithm(pop_size, elitism_size, norm_factor, mutation_prob, max_iteration, city_coords):
    """Perform a genetic algorithm (GA) to solve the Travelling Salesman Problem for a given group of cities.
    
    Args:
        pop_size (int): Number of routes within the population.
        elitism_size (float): Number in range [0,1] representing the top proportion of solutions to take forward.
        norm_factor - TODO
        mutation_prob (float): Probability with which to perform a mutation.
        max_iteration (int): Maximum number of iterations of the algorithm to perform.
        city_coords (dict): Dictionary containing the number of the node as key, and an (x, y) coordinate as value e.g. {1 : (34, 56)}.
    
    Returns:
        list: List of point numbers composing the best route found.
    """
    pop = generate_rand_population(pop_size, city_coords)

    for i in tqdm(range(0, max_iteration)):        
        pop = produce_next_gen(pop, pop_size, elitism_size, norm_factor, mutation_prob, city_coords)
        # print("Completed iteration " + str(iteration) + "/" + str(max_iteration), end="\r")
        if i % 10 == 0:
            print(" Best length : " + str(calc_route_length(pop[0], city_coords)))

    sol_best = rank_routes(pop, city_coords)[0]

    return sol_best

def tune_parameters_ga(min_pop_size, pop_step, pop_iter, min_elite, elite_step, elite_iter, min_norm, norm_step, norm_iter, min_mut, mut_step, mut_iter, max_iteration, city_coords):
    pop_size = min_pop_size
    elitism_size = min_elite
    norm_factor = min_norm 
    mutation_prob = min_mut

    (best_length, best_pop_size, best_elitism_size, best_norm_factor, best_mutation_prob) = (float("inf"), pop_size, elitism_size, norm_factor, mutation_prob)

    i = 0
    for i in tqdm(range(0, pop_iter)):
        j = 0
        elitism_size = min_elite
        for j in tqdm(range(0, elite_iter)):
            k = 0
            norm_factor = min_norm
            for k in tqdm(range(0, norm_iter)):
                m = 0
                mutation_prob = min_mut
                for m in tqdm(range(0, mut_iter)):

                    solution = perform_genetic_algorithm(pop_size, elitism_size, norm_factor, mutation_prob, max_iteration, city_coords)
                    length = calc_route_length(solution, city_coords)
                    if length < best_length:
                        (best_length, best_pop_size, best_elitism_size, best_norm_factor, best_mutation_prob) = (length, pop_size, elitism_size, norm_factor, mutation_prob)

                    # print("Completed run " + str(i) + "/" + str(pop_iter-1) + ", " + str(j) + "/" + str(elite_iter-1) + ", " + str(k) + "/" + str(norm_iter-1) + ", " + str(m) + "/" + str(mut_iter-1))
                
                    mutation_prob = mutation_prob + mut_step
                norm_factor = norm_factor + norm_step
            elitism_size = elitism_size + elite_step
        pop_size = pop_size + pop_step
        
    print("\n")
    return (best_length, best_pop_size, best_elitism_size, best_norm_factor, best_mutation_prob)

def do_ga_runs(pop_size, elitism_size, norm_factor, mutation_prob, max_iteration, city_coords):

    solutions = []
    lengths = []
    cumulative_lengths = 0

    for i in range(0, max_runs):
        sol = perform_genetic_algorithm(pop_size, elitism_size, norm_factor, mutation_prob, max_iteration, city_coords)
        solutions.append(sol)
        length = calc_route_length(sol, city_coords)
        lengths.append(length)
        cumulative_lengths = cumulative_lengths + length

    average_length = cumulative_lengths / max_runs
    standard_dev = statistics.stdev(lengths)

    return (solutions, lengths, average_length, standard_dev)
