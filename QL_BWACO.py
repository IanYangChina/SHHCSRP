"""------QL_BWACO.py------"""

import numpy as np
import pandas as pd
import copy
import math
import random


# Basic classes and input data
class Nurses:
    # Use '__slots__' to save memory
    __slots__ = ['l', 's', 'tt', 'twt', 'avg_w', 'r', 'sd', 'aT', 'ws']

    def __init__(self, label, skill):
        self.l = label
        self.s = skill
        # Total workload
        self.tt = 0
        # Total waiting time
        self.twt = 0
        # Average waiting time by job
        self.avg_w = 0
        # Visiting targets in line
        self.r = []
        # Fulfilled demand numbers by level
        self.sd = [0, 0, 0]
        # Elements: [[node label, arrival time, waiting time, service time, travel time to the next node],[]]
        self.aT = []
        # Waiting time at each target
        self.ws = []

    def clear(self):
        self.tt = 0
        self.twt = 0
        self.avg_w = 0
        self.r = []
        self.sd = [0, 0, 0]
        self.aT = []
        self.ws = []


class Jobs:
    # Use '__slots__' to save memory
    __slots__ = ['l', 'e', 'lv', 'c', 'twb', 'twe']

    def __init__(self, label, elder, level, coordinate_x, coordinate_y, coordinate_z,
                 time_window_begin, time_window_end):
        self.l = label
        self.e = elder
        self.lv = level
        self.c = [coordinate_x, coordinate_y, coordinate_z]
        self.twb = time_window_begin
        self.twe = time_window_end


def get_data(f_index, f_scale, jobs):
    # Setup statement
    file = './Elders_' + f_index + '.xlsx'

    # Temporal lists
    elders_index = []
    job_num = []
    job_level = []
    elder_location = []
    job_coordinate = np.zeros((f_scale, 3))
    time_window = np.zeros((f_scale, 2))

    # Read out column 'JobNum', 'Indexes', and 'JobLevel' from excel file
    excel = pd.read_excel(file, sheet_name='Sheet1')
    job_num.append(0)
    job_num += (list(copy.deepcopy(excel['JobNum'].values)))
    elders_index.append(0)
    elders_index += (list(copy.deepcopy(excel['Indexes'].values)))
    job_level.append(0)
    job_level += (list(copy.deepcopy(excel['JobLevel'].values)))

    # The first job is defined as the depot with coordinate (125, 125, 0)
    job_coordinate[0][0] = 125
    job_coordinate[0][1] = 125
    job_coordinate[0][2] = 0
    time_window[0][0] = 0.00
    time_window[0][1] = 480.00

    # Read out coordinates and time windows
    xyz = np.vstack((copy.deepcopy(excel['X'].values), copy.deepcopy(excel['Y'].values), copy.deepcopy(excel['Z'].values)))
    for i in range(len(xyz[0])):
        job_coordinate[i+1][0] = xyz[0][i]
        job_coordinate[i+1][1] = xyz[1][i]
        job_coordinate[i+1][2] = xyz[2][i]
    tw = np.vstack((copy.deepcopy(excel['TWB'].values), copy.deepcopy(excel['TWE'].values)))
    for i in range(len(tw[0])):
        time_window[i+1][0] = tw[0][i]
        time_window[i+1][1] = tw[1][i]

    # Read out locations labelled by elders for computing distance matrix
    lo = []
    for i in range(f_scale):
        lo.append([elders_index[i], job_coordinate[i][0], job_coordinate[i][1], job_coordinate[i][2]])
    for i in range(f_scale):
        if lo[i] in elder_location:
            continue
        else:
            elder_location.append(lo[i])

    # Build job classes and stack them into a list
    for fs in range(f_scale):
        jobs.append(
            Jobs(fs, elders_index[fs], job_level[fs], job_coordinate[fs][0], job_coordinate[fs][1],
                 job_coordinate[fs][2], time_window[fs][0], time_window[fs][1]))

    # Build distance matrix and return it
    num = len(elder_location)
    distance = np.zeros((num, num))
    for i in range(num):
        for j in range(num):
            hD = math.sqrt(pow((elder_location[i][1] - elder_location[j][1]), 2) + pow(
                (elder_location[i][2] - elder_location[j][2]), 2))
            if hD == 0:
                distance[i][j] = distance[j][i] = 9.6 * abs(elder_location[i][3] - elder_location[j][3])
            else:
                distance[i][j] = distance[j][i] = hD + elder_location[i][3] + elder_location[j][3]
    return distance


# QL realization
def state_identification(target_set, skill_set):
    a = len(skill_set)
    dn = count_demand_num(target_set)
    if a != 0:
        d1 = dn[0]
        d2 = dn[1]
        d3 = dn[2]
        d = sum(dn)
        if d != 0:
            if d1 >= d2 >= d3:
                return 0    # 0-d1>=d2>=d3
            elif d1 >= d3 >= d2:
                return 1   # 1-d1>=d3>=d2
            elif d2 >= d1 >= d3:
                return 2   # 2-d2>=d1>=d3
            elif d2 >= d3 >= d1:
                return 3   # 3-d2>=d3>=d1
            elif d3 >= d1 >= d2:
                return 4   # 4-d3>=d1>=d2
            elif d3 >= d2 >= d1:
                return 5   # 5-d3>=d2>=d1
        else:
            return 6   # 6-no more demands
    else:
        return 7   # 7-no more nurses


def state_identify(current_state_old, revisit):
    states = [[0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
              [3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
              [6, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
              [9, 10, 11, 11 ,11 ,11 ,11, 11, 11, 11, 11, 11 ,11 ,11 ,11, 11, 11, 11, 11, 11, 11],
              [12, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14],
              [15, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17]]
    if current_state_old == 6:
        return 18   # no more demands
    elif current_state_old == 7:
        return 19   # no more nurses
    else:
        return states[current_state_old][revisit[current_state_old]]


def action_taking(state, q_matrix, skill_set, greedy):
    # Generate a constant randomly
    g = random.uniform(0, 1)
    if g < greedy:
        # Act according to maximum q value
        q_values = copy.deepcopy(q_matrix[state])
        s1 = np.argmax(q_values)
        if s1 + 1 in skill_set:
            skill_set.remove(s1 + 1)
            return s1 + 1
        else:
            q_values[s1] = -1
            s2 = np.argmax(q_values)
            if s2 + 1 in skill_set:
                skill_set.remove(s2 + 1)
                return s2 + 1
            else:
                q_values[s2] = -1
                s3 = np.argmax(q_values)
                skill_set.remove(s3 + 1)
                return s3 + 1
    else:
        # Act randomly as exploration
        skill = copy.deepcopy(skill_set)
        random.shuffle(skill)
        action = skill[0]
        skill_set.remove(action)
        return action


def count_demand_num(target_set):
    d = [0, 0, 0]
    for j in range(len(target_set)):
        if target_set[j].lv == 1:
            d[0] += 1
        if target_set[j].lv == 2:
            d[1] += 1
        if target_set[j].lv == 3:
            d[2] += 1
    return d


def q_learning(q_nurses_list, target_set, skill_set,
               q_matrix, learning_rate_para, discount_para, greedy_para,
               # ACO variables
               ccp_acquaintance_increment, ccp_alpha, ccp_beta, ccp_waiting_limit, ccp_workload_limit,
               aco_alpha, aco_beta, aco_rho, aco_ant_number, changeable_pheromone_matrix, aco_increment,
               e_walk_speed, changeable_preference_matrix, e_init_distance_matrix, e_service_time_mean,
               depot):
    # Start a new QL episode
    q_sub_total_workload = 0
    q_sub_total_waiting = 0
    q_sub_total_reward = 0
    q_reward_per_action = []
    n = 0
    revisit = [0, 0, 0, 0, 0, 0]
    absorb = False
    training_input = []
    training_label = []
    while not absorb:
        # Data recording variable
        dd = count_demand_num(target_set)
        ds = [str(skill_set).count('1'), str(skill_set).count('2'), str(skill_set).count('3')]
        # Identify current state based solely on demand size-relationships
        current_state = state_identification(target_set, skill_set)
        # Identify current state with revisit times
        current_state_new = state_identify(current_state, revisit)
        revisit[current_state] +=1
        # Take action according to e-greedy
        chosen_skill_new = action_taking(current_state_new, q_matrix, skill_set, greedy_para)
        # Create nurse object in a list sequentially
        q_nurses_list.append(Nurses(n, chosen_skill_new))
        # Collect targets according to skill demand match rule
        sd_matched_targets = []
        for aj in range(len(target_set)):
            if target_set[aj].lv <= q_nurses_list[n].s:
                sd_matched_targets.append(target_set[aj])

        # Build route by ACO algorithm
        aco(ccp_acquaintance_increment, ccp_alpha, ccp_beta, ccp_waiting_limit, ccp_workload_limit,
            aco_alpha, aco_beta, aco_rho, aco_ant_number, changeable_pheromone_matrix, aco_increment,
            e_walk_speed, changeable_preference_matrix, e_init_distance_matrix, e_service_time_mean,
            q_nurses_list[n], sd_matched_targets, depot)

        # Remove fulfilled demands
        for o in range(len(q_nurses_list[n].r)):
            if q_nurses_list[n].r[o].l != 0:
                for b in range(len(target_set)):
                    if target_set[b].l == q_nurses_list[n].r[o].l:
                        target_set.remove(target_set[b])
                        break

        # Calculate fulfilled demands
        # Reward function updated on 19/09/2018, equals to the amount of fulfilled demands
        # Reward function updated on 12/01/2019, equals to a function of waiting time at non-absorbing states
        # Reward function updated on 01/02/2019,
        #   equals to a function of waiting time and fulfilled demands at non-absorbing states
        # Reward function updated on 02/12/2019, add a condition set it to 0 if no demands were fulfilled
        # Reward function updated on 27/03/2019, balance between the influences of waiting time and fulfilled demands
        if sum(q_nurses_list[n].sd) == 0:
            reward = 0
        else:
            reward = (1000 * sum(q_nurses_list[n].sd)) / (20 * (q_nurses_list[n].twt + 10)) + sum(q_nurses_list[n].sd)

        next_state = state_identification(target_set, skill_set)
        next_state_new = state_identify(next_state, revisit)

        # Calculate q value
        next_max_q = np.max(q_matrix[next_state_new])
        q_value = (1 - learning_rate_para) * q_matrix[current_state_new][chosen_skill_new - 1]\
                  + learning_rate_para * (reward + discount_para * next_max_q)
        # Update Q-matrix
        q_matrix[current_state_new][chosen_skill_new - 1] = float('%.2f' % q_value)
        # Record training data
        training_input.append([dd[0], dd[1], dd[2], ds[0], ds[1], ds[2], chosen_skill_new])
        training_label.append(float('%.2f' % q_value))

        # Accumulate total workload
        q_sub_total_workload += q_nurses_list[n].tt
        q_sub_total_waiting += q_nurses_list[n].twt

        # Check if the next state is an absorbing state
        # If so, update the q value of the absorbing states and end loop
        if next_state_new in [18, 19]:
            # reach absorbing state, set ending flag to 1
            absorb = True
            # Calculate reward as a function of remaining demands and update q matrix for all absorbing states
            rde = count_demand_num(target_set)
            reward_e = 1000 / (sum(rde) + 2)
            reward += reward_e
            q_value_e = (1 - learning_rate_para) * q_matrix[next_state_new][0] + learning_rate_para * reward_e
            q_matrix[next_state_new] = float('%.2f' % q_value_e)

            q_sub_total_reward += reward
            q_reward_per_action.append(float('%.2f' % reward))
            # Record training data
            dse = [str(skill_set).count('1'), str(skill_set).count('2'), str(skill_set).count('3')]
            training_input.append([rde[0], rde[1], rde[2], dse[0], dse[1], dse[2], 0])
            training_label.append(float('%.2f' % q_value_e))
            break

        q_sub_total_reward += reward
        q_reward_per_action.append(float('%.2f' % reward))

        # Nurse label plus 1
        n += 1

    # Return total workload and waiting for this episode
    return [q_sub_total_workload, q_sub_total_waiting, q_sub_total_reward, q_reward_per_action, training_input, training_label]


def q_learning_test(q_nurses_list, target_set, skill_set,
                    q_matrix, greedy_para,
                    # ACO variables
                    ccp_acquaintance_increment, ccp_alpha, ccp_beta, ccp_waiting_limit, ccp_workload_limit,
                    aco_alpha, aco_beta, aco_rho, aco_ant_number, changeable_pheromone_matrix, aco_increment,
                    e_walk_speed, changeable_preference_matrix, e_init_distance_matrix, e_service_time_mean,
                    depot):
    # Start a new QL episode
    q_sub_total_workload = 0
    q_sub_total_waiting = 0
    q_sub_total_reward = 0
    q_reward_per_action = []
    n = 0
    revisit = [0, 0, 0, 0, 0, 0]
    absorb = False
    while not absorb:
        # Identify current state
        demand_size_relation = state_identification(target_set, skill_set)
        current_state = state_identify(demand_size_relation, revisit)
        revisit[demand_size_relation] +=1
        # Take action according to e-greedy
        chosen_skill = action_taking(current_state, q_matrix, skill_set, greedy_para)
        # Create nurse object in a list sequentially
        q_nurses_list.append(Nurses(n, chosen_skill))
        # Collect targets according to skill demand match rule
        sd_matched_targets = []
        for aj in range(len(target_set)):
            if target_set[aj].lv <= q_nurses_list[n].s:
                sd_matched_targets.append(target_set[aj])

        # Build route by ACO algorithm
        aco(ccp_acquaintance_increment, ccp_alpha, ccp_beta, ccp_waiting_limit, ccp_workload_limit,
            aco_alpha, aco_beta, aco_rho, aco_ant_number, changeable_pheromone_matrix, aco_increment,
            e_walk_speed, changeable_preference_matrix, e_init_distance_matrix, e_service_time_mean,
            q_nurses_list[n], sd_matched_targets, depot)

        # Remove fulfilled demands
        for o in range(len(q_nurses_list[n].r)):
            if q_nurses_list[n].r[o].l != 0:
                for b in range(len(target_set)):
                    if target_set[b].l == q_nurses_list[n].r[o].l:
                        target_set.remove(target_set[b])
                        break

        if sum(q_nurses_list[n].sd) == 0:
            reward = 0
        else:
            reward = (1000 * sum(q_nurses_list[n].sd)) / (20 * (q_nurses_list[n].twt + 10)) + sum(q_nurses_list[n].sd)

        next_demand_size_relation = state_identification(target_set, skill_set)
        next_state = state_identify(next_demand_size_relation, revisit)

        # Accumulate total workload, waiting time and return
        q_sub_total_workload += q_nurses_list[n].tt
        q_sub_total_waiting += q_nurses_list[n].twt

        if next_state in [18, 19]:
            # reach absorbing state, set ending flag to 1
            absorb = True
            # Calculate reward as a function of remaining demands and update q matrix for all absorbing states
            remaining_demands_e = count_demand_num(target_set)
            reward_e = 1500 / (sum(remaining_demands_e) + 2)
            reward += reward_e
            q_sub_total_reward += reward
            q_reward_per_action.append(float('%.2f' % reward))
            break

        q_sub_total_reward += reward
        q_reward_per_action.append(float('%.2f' % reward))

        # Nurse label plus 1
        n += 1

    # Return total workload and waiting for this episode
    return [q_sub_total_workload, q_sub_total_waiting, q_sub_total_reward, q_reward_per_action]


# ACO realization
def collect_feasible_targets(visiting_list, distance_matrix, walk_speed, waiting_limit, current_job, current_time):
    ft = []
    for j in range(len(visiting_list)):
        distance = distance_matrix[current_job.e][visiting_list[j].e]
        travel = distance / walk_speed
        arrival = current_time + travel
        # Arrival time must be earlier than the upper bound
        # and later than the maximum waiting time + lower bound
        if arrival < visiting_list[j].twe:
            if (arrival + waiting_limit) >= visiting_list[j].twb:
                ft.append(visiting_list[j])
                continue
        else:
            continue
    return ft


def choose_target_deterministically(pr):
    p_axi = []
    for pta in range(len(pr)):
        p_axi.append(pr[pta][2])
    max_ind = np.argmax(p_axi)
    return max_ind


def choose_target_randomly(pr):
    p_coor = 0
    p_axi = [0]

    for pta in range(len(pr)):
        p_coor += pr[pta][2]
        p_axi.append(p_coor)
    # generate a random value
    ran_var = random.uniform(0, p_coor)
    search = 1

    while ran_var > 0:
        if ran_var <= p_axi[search]:
            break
        else:
            search += 1

    return search-1


def calculate_transition_probability(feasible_targets, current_time, distance_matrix, current_job, walk_speed,
                                     ant_path_table, visiting_list, pheromone_matrix,
                                     alpha_aco_p, beta_aco_p,
                                     depot):
    # Count feasible targets
    # =0: return depot as the next target
    # =1: return it as the next target
    # >2: return the target chosen according to probability transition function
    if (len(feasible_targets)) == 0:
        # No feasible targets, end routing
        current_time += (distance_matrix[current_job.e][depot.e]) / walk_speed
        ant_path_table.append(depot)
        return depot
    elif len(feasible_targets) == 1:
        # Only one feasible target, choose it and update route
        current_time += (distance_matrix[current_job.e][feasible_targets[0].e]) / walk_speed
        ant_path_table.append(feasible_targets[0])
        # Remove chosen target from visiting list
        for v in range(len(visiting_list)):
            if visiting_list[v].l == feasible_targets[0].l:
                visiting_list.remove(visiting_list[v])
                return feasible_targets[0]
    else:
        # More than 1 feasible targets, calculate transition probabilities
        pr = []
        pD = 0
        for pdd in range(len(feasible_targets)):
            target_waiting = feasible_targets[pdd].twb - \
                             current_time + distance_matrix[current_job.e][feasible_targets[pdd].e] / walk_speed
            if target_waiting < 0:
                yitaD = 10
            else:
                yitaD = 1000 / (target_waiting + 100)
            pD += pow((pheromone_matrix[current_job.e][feasible_targets[pdd].e]), alpha_aco_p) \
                  * pow(yitaD, beta_aco_p)
        for pt in range(len(feasible_targets)):
            target_waiting = feasible_targets[pt].twb - \
                             current_time + distance_matrix[current_job.e][feasible_targets[pt].e] / walk_speed
            if target_waiting < 0:
                yitaU = 10
            else:
                yitaU = 1000 / (target_waiting + 100)
            pU = pow((pheromone_matrix[current_job.e][feasible_targets[pt].e]), alpha_aco_p) \
                 * pow(yitaU, beta_aco_p)
            pT = pU / pD
            pr.append([current_job, feasible_targets[pt], pT])
        # Choose target randomly and update route
        target_index = choose_target_randomly(pr)
        ant_path_table.append(pr[target_index][1])
        current_time += (distance_matrix[current_job.e][pr[target_index][1].e]) / walk_speed
        # Remove chosen target from visiting list
        for v in range(len(visiting_list)):
            if visiting_list[v].l == pr[target_index][1].l:
                visiting_list.remove(visiting_list[v])
                break
        return pr[target_index][1]


def update_pheromone(best_path, worst_path, pheromone_matrix, rho_aco_p, distance_matrix, x):
    # update pheromone according to Best-Worst rule
    pheromone_matrix *= (1 - rho_aco_p)
    for bP in range(len(best_path)):
        if (bP + 1) == len(best_path):
            break
        else:
            pheromone_matrix[best_path[bP].e][best_path[bP + 1].e] += x
    """
    for wP in range(len(worst_path)):
        if (wP + 1) == len(worst_path):
            break
        else:
            if x != "Distance":
                pheromone_matrix[worst_path[wP].e][worst_path[wP + 1].e] \
                    = (1 - rho_aco_p) * pheromone_matrix[worst_path[wP].e][worst_path[wP + 1].e] - x
            else:
                pheromone_matrix[worst_path[wP].e][worst_path[wP + 1].e] \
                    = (1 - rho_aco_p) * pheromone_matrix[worst_path[wP].e][worst_path[wP + 1].e] \
                      - distance_matrix[worst_path[wP].e][worst_path[wP + 1].e]
    """


def aco(acquaintance_increment, alpha_model_p, beta_model_p, waiting_limit, workload,
        alpha_aco_p, beta_aco_p, rho_aco_p, ant_num, pheromone_matrix, pheromone_increment,
        walk_speed, preference_matrix, distance_matrix, service_time_mean,
        nurse, sd_matched_targets, depot):
    # Output variables
    time_schedule_final = []
    shortest_time = 0
    waiting_time = 0
    waiting_sequence = []
    best_path = []
    # Temporal variables
    worst_path = []
    ccp_best_objective = 0
    ccp_worst_objective = 0

    for ant in range(ant_num):
        # Initialization: depot, time, waiting time, workload
        current_job = depot
        current_time = 0
        current_waiting = 0
        current_waiting_sequence = [0]
        current_workload = 0
        # Lists of timings of visits
        # Elements: [[node label, arrival time, waiting time, service time, travel time to the next node], ...]
        ts = 0
        time_schedule = [[0, 0]]
        ant_path_table = [depot]
        # Initialize visiting list and preference matrix
        visiting_list = copy.deepcopy(sd_matched_targets)
        current_preference = copy.deepcopy(preference_matrix)

        # Build routes
        while current_workload <= workload:
            # Read out service time mean value and preference value
            st_mean = service_time_mean[nurse.s][current_job.lv]
            preference_factor = copy.deepcopy(current_preference[current_job.e][nurse.l])

            # Inspect waiting and record sub-arrival time
            if current_time < current_job.twb:
                current_waiting += (current_job.twb - current_time)
                current_waiting_sequence.append(float('%.2f' % (current_job.twb - current_time)))
                time_schedule[ts].append(float('%.2f' % (current_job.twb - current_time)))
                current_time = copy.deepcopy(current_job.twb)
            else:
                current_waiting_sequence.append(0)
                time_schedule[ts].append(0)

            # Compute arrival time as predicted workload when going back to depot at current position
            # Then check if overwork occurs
            current_workload = current_time + (preference_factor * st_mean)\
                               + beta_model_p + (distance_matrix[current_job.e][depot.e]) / walk_speed
            if current_workload >= workload:
                # Overwork predicted, stop routing
                # Set depot as the next target and record arrival time
                ant_path_table.append(depot)
                current_time = copy.deepcopy(current_workload)
                time_schedule[ts].append((preference_factor * st_mean) + alpha_model_p)
                time_schedule[ts].append((distance_matrix[current_job.e][depot.e]) / walk_speed)
                time_schedule.append([0, current_workload, 0, 0, 0])
                break
            else:
                # Continue routing
                # Add up service time
                if ts == 0:
                    current_time += (preference_factor * st_mean)
                    time_schedule[ts].append(preference_factor * st_mean)
                else:
                    current_time += (preference_factor * st_mean) + alpha_model_p
                    time_schedule[ts].append((preference_factor * st_mean) + alpha_model_p)

            # Search for targets satisfying the time window constraint
            feasible_targets = collect_feasible_targets(visiting_list, distance_matrix, walk_speed, waiting_limit,
                                                        current_job, current_time)
            # Count feasible targets, calculate transition probabilities and choose target
            chosen_target = calculate_transition_probability(feasible_targets, current_time, distance_matrix,
                                                             current_job, walk_speed, ant_path_table,
                                                             visiting_list, pheromone_matrix, alpha_aco_p, beta_aco_p,
                                                             depot)
            if chosen_target.l == 0:
                time_schedule[ts].append((distance_matrix[current_job.e][depot.e]) / walk_speed)
                current_time += ((distance_matrix[current_job.e][depot.e]) / walk_speed)
                time_schedule.append([0, current_time, 0, 0, 0])
                # No feasible target, back to depot, stop routing
                break
            else:
                current_time += ((distance_matrix[current_job.e][chosen_target.e]) / walk_speed)
                time_schedule[ts].append((distance_matrix[current_job.e][chosen_target.e]) / walk_speed)
                time_schedule.append([chosen_target.e, current_time])
                # Feasible target chosen, continue
                current_job = chosen_target
                # Revise preference
                if current_preference[chosen_target.e][nurse.l] > 0.7:
                    current_preference[chosen_target.e][nurse.l] -= acquaintance_increment
                ts += 1
                continue


        # Calculate fulfilled demands
        fulfilled_demand = copy.deepcopy(len(ant_path_table) - 2)
        # Record the best and worst solution according to the CCP objective
        if fulfilled_demand == 0:
            # no fulfilled demand
            if len(best_path) == 0:
                best_path = copy.deepcopy(ant_path_table)
                # worst_path = copy.deepcopy(ant_path_table)
        else:
            # record current PPM objective: total waiting time
            ccp_objective = copy.deepcopy(current_waiting)
            if ant == 0:
                # first iteration, record best CCP objective, worst PPM objective, working time,
                # waiting time, best route, and worst route
                ccp_best_objective = copy.deepcopy(ccp_objective)
                ccp_worst_objective = copy.deepcopy(ccp_objective)
                shortest_time = copy.deepcopy(current_time)
                waiting_time = copy.deepcopy(current_waiting)
                waiting_sequence = copy.deepcopy(current_waiting_sequence)
                best_path = copy.deepcopy(ant_path_table)
                # worst_path = copy.deepcopy(ant_path_table)
                time_schedule_final = copy.deepcopy(time_schedule)
            else:  # not first iteration
                if ccp_best_objective > ccp_objective:  # find the best one
                    ccp_best_objective = copy.deepcopy(ccp_objective)
                    shortest_time = copy.deepcopy(current_time)
                    waiting_time = copy.deepcopy(current_waiting)
                    waiting_sequence = copy.deepcopy(current_waiting_sequence)
                    best_path = copy.deepcopy(ant_path_table)
                    time_schedule_final = copy.deepcopy(time_schedule)
                    """
                    elif ccp_worst_objective < ccp_objective:  # find the worst one
                        ccp_worst_objective = copy.deepcopy(ccp_objective)
                        worst_path = copy.deepcopy(ant_path_table)
                    """
                else:
                    continue

        # update pheromone according to Best-Worst rule
        update_pheromone(best_path, worst_path, pheromone_matrix, rho_aco_p, distance_matrix, pheromone_increment)

    # update route
    nurse.tt = copy.deepcopy(shortest_time)
    nurse.aT = copy.deepcopy(time_schedule_final)
    nurse.twt = copy.deepcopy(waiting_time)
    nurse.ws = copy.deepcopy(waiting_sequence)
    for o in range(len(best_path)):
        nurse.r.append(best_path[o])
        if best_path[o].lv == 1:
            nurse.sd[0] += 1
        elif best_path[o].lv == 2:
            nurse.sd[1] += 1
        elif best_path[o].lv == 3:
            nurse.sd[2] += 1
    if sum(nurse.sd) != 0:
        nurse.avg_w = float('%.2f' % (copy.deepcopy(nurse.twt / sum(nurse.sd))))
