from route_functions import *


def generate_population(pop_size, city_coords):
    pop = []

    for i in range(0, pop_size):
        pop.append(generate_rand_route(city_coords))

    return pop

def rank_routes(pop, city_coords):
    ids_and_lengths = {}

    for i in range(0, len(pop)):
        ids_and_lengths[i] = calc_route_length(pop[i], city_coords)
    ranked_ids = {k: v for k, v in sorted(ids_and_lengths.items(), key=lambda item: item[1])}

    ranked_routes = []

    for id in ranked_ids.keys():
        ranked_routes.append(pop[id])

    return ranked_routes

def select_parents(pop, pop_size, elitism_size):
    # Simple truncation/elitism selection
    group_size = math.floor(pop_size * elitism_size)

    return pop[0:group_size]

def breed_route(parent1, parent2):
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
    mutated_pop = []

    for route in child_routes:
        mutant = mutate_route(route, mutation_prob)
        mutated_pop.append(mutant)

    return mutated_pop

def produce_next_gen(pop, pop_size, elitism_size, mutation_prob, city_coords):
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
    pop = generate_population(pop_size, city_coords)

    iteration = 0

    while iteration < max_iteration:        
        pop = produce_next_gen(pop, pop_size, elitism_size, mutation_prob, city_coords)
        iteration = iteration + 1

    sol_best = rank_routes(pop, city_coords)[0]

    return sol_best