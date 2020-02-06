from route_functions import *


def get_energy(solution, city_coords):
    return calc_route_length(solution, city_coords)
    
def get_rand_neighbour(solution):
    # max_pos = len(solution) - 1
    # rand_pos_1 = random.randint(2, max_pos)
    # rand_pos_2 = random.randint(0, max_pos - rand_pos_1)

    # neighbour = solution.copy()
    # neighbour[rand_pos_1 : (rand_pos_1 + rand_pos_2)] = reversed(neighbour[rand_pos_1 : (rand_pos_1 + rand_pos_2)])

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

def calc_acceptance_prob(energy_cur, energy_new, temp):
    if energy_new < energy_cur:
        return 1
    else:
        return math.exp((energy_cur - energy_new) / temp)

def calc_new_temperature(temp, cooling_rate):
    return temp * cooling_rate

def perform_simulated_annealing(temp_initial, cooling_rate, max_iteration, city_coords):
    sol_initial = generate_rand_route(city_coords)
    sol_cur = sol_initial
    energy_cur = get_energy(sol_cur, city_coords)
    sol_best = sol_cur
    energy_best = energy_cur

    temp = temp_initial
    iteration = 0

    while temp > 0 and iteration < max_iteration:
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
