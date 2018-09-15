import time
import copy
import numpy as np
import pandas as pd
import xlwt
import matplotlib.pyplot as plt
import QL_BWACO
import os
os.getcwd()


class Solution:
    __slots__ = ['nl', 'on', 'ev', 'rd', 'fd_sum',
                 'a_wait_j', 'a_wait_n', 't_wait', 'a_work_j', 'a_work_n', 't_work',
                 'fig_t', 'info_to_excel']

    def __init__(self):
        # Nurse list
        self.nl = []
        # Occupied nurse number
        self.on = 0
        # Evaluation function value
        self.ev = -1000
        # Remaining demand number
        self.rd = []
        # Fulfilled demand number
        self.fd_sum = 0
        # Average waiting time of fulfilled job
        self.a_wait_j = 0
        # Average waiting time of occupied nurse
        self.a_wait_n = 0
        # Total waiting time
        self.t_wait = 0
        # Average service time of job
        self.a_work_j = 0
        # Average workload of occupied nurse
        self.a_work_n = 0
        # Total workload of occupied nurse
        self.t_work = 0
        # Figure test
        self.fig_t = ''
        # Information to excel
        self.info_to_excel = []

    def calculate(self, q_learning_result, remaining_demand):
        self.t_work = float('%.2f' % (copy.deepcopy(q_learning_result[0])))
        self.t_wait = float('%.2f' % (copy.deepcopy(q_learning_result[1])))
        self.rd = QL_BWACO.count_demand_num(copy.deepcopy(remaining_demand))
        self.fd_sum = file_scale - sum(self.rd) - 1
        # Calculate occupied nurses
        for o in range(len(self.nl)):
            if len(self.nl[o].r) > 2:
                self.on += 1
        if self.on != 0:
            self.a_wait_j = float('%.2f' % (self.t_wait / self.fd_sum))
            self.a_wait_n = float('%.2f' % (self.t_wait / self.on))
            self.a_work_j = float('%.2f' % (self.t_work / self.fd_sum))
            self.a_work_n = float('%.2f' % (self.t_work / self.on))
            self.ev = float('%.2f' % (self.fd_sum - 0.25 * self.t_wait))

    def get_solution_info(self):
        indexes = []
        skill_level = []
        workload = []
        waiting = []
        avg_waiting = []
        route_e = []
        route_j = []
        fulfilled_d = []
        # Stack solution information by columns
        for i in range(len(self.nl)):
            indexes.append(self.nl[i].l)
            skill_level.append(self.nl[i].s)
            workload.append(self.nl[i].tt)
            waiting.append(self.nl[i].twt)
            avg_waiting.append(self.nl[i].avg_w)
            e = []
            j = []
            for r in range(len(self.nl[i].r)):
                e.append(self.nl[i].r[r].e)
                j.append(self.nl[i].r[r].l)
            route_e.append(e)
            route_j.append(j)
            fulfilled_d.append(self.nl[i].sd)
        # return organized information as a list
        self.info_to_excel = copy.deepcopy([indexes, skill_level, workload, waiting, avg_waiting, route_e, route_j, fulfilled_d])

    def save_solution_info(self, file_name):
        # save solution information into excel file
        exc = xlwt.Workbook()
        exc.add_sheet("Sheet1")
        exc.save(file_name)
        writer = pd.ExcelWriter(file_name, sheet_name='Sheet1')
        solution_df = pd.DataFrame({'Nurses': self.info_to_excel[0],
                                    'Skill': self.info_to_excel[1],
                                    'Workload': self.info_to_excel[2],
                                    'Waiting': self.info_to_excel[3],
                                    'Average waiting': self.info_to_excel[4],
                                    'Routes by elder': self.info_to_excel[5],
                                    'Routes by job': self.info_to_excel[6],
                                    'Fulfilled demands': self.info_to_excel[7]})
        solution_df.to_excel(writer, sheet_name='Sheet1', index=False)
        writer.save()

    def create_fig_text(self, running_time):
        if running_time != 0:
            self.fig_t = "\n" \
                         "Running time: " + str(float('%.2f' % running_time)) + " mins \n" \
                         "Results of best solution found:\n" \
                         "Solution evaluation value = " + str(self.ev) + "\n" \
                         "Solution avg workload of nurse = " + str(self.a_work_n) + "\n" \
                         "Solution avg waiting time of job = " + str(self.a_wait_j) + "\n" \
                         "Solution avg waiting time of nurse = " + str(self.a_wait_n) + "\n" \
                         "Solution total workload = " + str(self.t_work) + "\n" \
                         "Solution total waiting time = " + str(self.t_wait) + "\n" \
                         "Solution remaining demands = " + str(self.rd)
        else:
            self.fig_t = "\n" \
                         "Results of testing the trained agent:\n" \
                         "Solution evaluation value = " + str(self.ev) + "\n" \
                         "Solution avg workload of nurse = " + str(self.a_work_n) + "\n" \
                         "Solution avg waiting time of job = " + str(self.a_wait_j) + "\n" \
                         "Solution avg waiting time of nurse = " + str(self.a_wait_n) + "\n" \
                         "Solution total workload = " + str(self.t_work) + "\n" \
                         "Solution total waiting time = " + str(self.t_wait) + "\n" \
                         "Solution remaining demands = " + str(self.rd)


def plot(figure_name, y_label, x_label, x, y, figure_text):
    y_lower = min(y) - 5
    y_upper = max(y) + 5
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.xlim(0, x)
    plt.ylim(y_lower, y_upper)
    plt.plot(y)
    # Figure explanations
    plt.text(x-1, y_lower+1, figure_text, fontsize=9, va="baseline", ha="right")
    name = figure_name + '.png'
    plt.savefig(name, dpi=900, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':

    """------Instance representations------"""
    # Get elder data and distance matrix
    file_index = 'A'
    file_scale = 61
    e_jobs = []
    e_init_distance_matrix = QL_BWACO.get_data(file_index, file_scale, e_jobs)
    # Set up nurse resource
    e_nurses_skill = [1, 1, 2, 2, 3, 3, 3]
    # Build initial preference matrix
    e_random_pre = np.loadtxt("./initial_preference_ABC.csv", delimiter=",", skiprows=0)
    e_preference_matrix = np.row_stack((np.zeros((1, len(e_nurses_skill))), e_random_pre))
    # Skill demand matching parameter matrix
    e_service_time_mean = np.array([[0, 0, 0, 0],
                                    [0, 25, -1, -1],
                                    [0, 20, 30, -1],
                                    [0, 18, 20, 20]])
    e_walk_speed = 60

    """Parameter settings"""
    # learningRate discount greedy waiting
    para_ql = [[0.1, 0.9, 0.5, -0.01],
               [0.5, 0.9, 0.5, -0.01],
               [0.9, 0.9, 0.5, -0.01]]

    # initialPheromone alpha beta evaporation
    para_aco = [[20, 1, 1, 0.1],
                [20, 1, 1, 0.5],
                [20, 5, 1, 0.1],
                [20, 5, 1, 0.5],
                [20, 1, 5, 0.1],
                [20, 1, 5, 0.5]]
    # confidence levels:        100%  95%   90%   80%   70%   60%   50%
    # inverse standard normal:  3.09  1.65, 1.29  0.85  0.52  0.26  0
    # acquaintance_increment waiting workload alpha beta
    para_ccp = [[0.05, 10, 480, 1.29, 1.29],
                [0.05, 20, 480, 1.29, 1.29],
                [0.05, 30, 480, 1.29, 1.29],
                [0.05, 40, 480, 1.29, 1.29],
                [0.05, 50, 480, 1.29, 1.29]]

    """------Q Learning parameters------"""
    ql_learning_rate_para = para_ql[0][0]
    ql_discount_para = para_ql[0][1]
    ql_greedy_para = para_ql[0][2]
    ql_waiting_para = para_ql[0][3]

    ql_q_matrix = np.zeros((8, 3))
    ql_absorbing_state = [6, 7]

    """------ACO parameters------"""
    aco_alpha = 5
    aco_beta = 5
    aco_rho = 0.3
    aco_ant_number = 30
    aco_pheromone = 20
    aco_pheromone_matrix = np.ones((len(e_init_distance_matrix[0]), len(e_init_distance_matrix[0]))) * aco_pheromone

    """------CCP model parameters------"""
    ccp_acquaintance_increment = 0.01
    ccp_alpha = 1.29
    ccp_beta = 1.29
    ccp_waiting_limit = 40
    ccp_workload_limit = 480

    """---Solution Recording Variables---"""
    solution_final = Solution()

    """Figure data record"""
    axis_sub_evaluation_value_iter = []
    axis_avg_wait_nurse_iter = []
    axis_evaluation_value_iter = []

    """---Start iteration---"""
    # Record the starting time of the training process
    start = time.time()
    iter = 0
    iter_max = 500
    print('Current Experimental Instance is ' + file_index)
    print('Instance Scale: ' + str(file_scale-1))
    print('Start training...')
    while iter < iter_max:
        iter += 1
        # Solution objective for current sub-solution
        sub_solution = Solution()
        # Changeable list of targets
        available_targets = copy.deepcopy(e_jobs)
        # Delete depot
        available_targets.remove(available_targets[0])
        # Changeable preference, pheromone matrix and nurse's skill set
        changeable_preference_matrix = copy.deepcopy(e_preference_matrix)
        changeable_pheromone_matrix = copy.deepcopy(aco_pheromone_matrix)
        changeable_nurses_skill = copy.deepcopy(e_nurses_skill)
        # Start Q Learning process
        ql_result = QL_BWACO.q_learning(sub_solution.nl, available_targets, changeable_nurses_skill,
                                        ql_q_matrix, ql_learning_rate_para, ql_discount_para, ql_greedy_para,
                                        ql_waiting_para, ql_absorbing_state,
                                        # ACO variables
                                        ccp_acquaintance_increment, ccp_alpha, ccp_beta, ccp_waiting_limit,
                                        ccp_workload_limit,
                                        aco_alpha, aco_beta, aco_rho, aco_ant_number, changeable_pheromone_matrix,
                                        e_walk_speed, changeable_preference_matrix, e_init_distance_matrix,
                                        e_service_time_mean,
                                        e_jobs[0])
        # Calculate fulfilled and remaining demands
        sub_solution.calculate(ql_result, available_targets)
        # Update global solution according to evaluation value
        if solution_final.ev < sub_solution.ev:
            solution_final = copy.deepcopy(sub_solution)

        axis_avg_wait_nurse_iter.append(solution_final.a_wait_n)
        axis_evaluation_value_iter.append(solution_final.ev)
        axis_sub_evaluation_value_iter.append(sub_solution.ev)

    # Record the ending time of the training process
    end = time.time()
    rt = (end - start) / 60
    # Create figure text for the solution
    solution_final.create_fig_text(rt)
    print(solution_final.fig_t)
    # Save detailed information into an excel file
    solution_final.get_solution_info()
    solution_final.save_solution_info("final solution.xls")
    # Plot the figures
    plot("iter - evaluation value", "evaluation value", "iteration",
         iter_max - 1, axis_evaluation_value_iter, solution_final.fig_t)
    plot("iter - sub evaluation value", "sub ev", "iteration",
         iter_max - 1, axis_sub_evaluation_value_iter, solution_final.fig_t)
    # Show the trained Q matrix on the console
    print(np.around(ql_q_matrix, decimals=2))

    """Solution after training"""
    print("\nStart testing...")
    trained_q_matrix = copy.deepcopy(ql_q_matrix)
    print("Trained Q matrix:")
    test_nurses = copy.deepcopy(e_nurses_skill)
    test_targets = copy.deepcopy(e_jobs)
    test_preference_matrix = copy.deepcopy(e_preference_matrix)
    test_pheromone_matrix = copy.deepcopy(aco_pheromone_matrix)
    test_solution = Solution()
    test_ql_results = QL_BWACO.q_learning_test(test_solution.nl, test_targets, test_nurses,
                                               trained_q_matrix, 0, ql_absorbing_state,
                                               # Aco para
                                               ccp_acquaintance_increment, ccp_alpha, ccp_beta, ccp_waiting_limit,
                                               ccp_workload_limit,
                                               aco_alpha, aco_beta, aco_rho, aco_ant_number, test_pheromone_matrix,
                                               e_walk_speed, test_preference_matrix, e_init_distance_matrix,
                                               e_service_time_mean,
                                               e_jobs[0])
    test_solution.calculate(test_ql_results, test_targets)
    test_solution.create_fig_text(0)
    print(test_solution.fig_t)
    test_solution.get_solution_info()
    test_solution.save_solution_info("test solution.xls")
