import time
import copy
import xlwt
import os
import numpy as np
import pandas as pd
import QL_BWACO
import Plot
os.getcwd()


class Solution:
    __slots__ = ['nl', 'on', 'ev', 'rd', 'fd_sum', 'rewards',
                 'a_wait_j', 'a_wait_n', 't_wait', 'wait_s', 'wait_sl',
                 'a_work_j', 'a_work_n', 't_work', 'wait_d_work',
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
        # Rewards gained by each nurse
        self.rewards = []
        # Average waiting time of fulfilled job
        self.a_wait_j = 0
        # Average waiting time of occupied nurse
        self.a_wait_n = 0
        # Total waiting time
        self.t_wait = 0
        # Waiting sequence and labels
        self.wait_s = []
        self.wait_sl = []
        # Average service time of job
        self.a_work_j = 0
        # Average workload of occupied nurse
        self.a_work_n = 0
        # Total workload of occupied nurse
        self.t_work = 0
        # Percentage of waiting in workload
        self.wait_d_work = 0
        # Figure test
        self.fig_t = ''
        # Information to excel
        self.info_to_excel = []

    def calculate(self, q_learning_result, remaining_demand):
        self.t_work = float('%.2f' % (copy.deepcopy(q_learning_result[0])))
        self.t_wait = float('%.2f' % (copy.deepcopy(q_learning_result[1])))
        self.ev = float('%.2f' % (copy.deepcopy(q_learning_result[2])))
        self.rewards = copy.deepcopy(q_learning_result[3])
        self.rd = QL_BWACO.count_demand_num(copy.deepcopy(remaining_demand))
        self.fd_sum = file_scale - sum(self.rd) - 1
        self.wait_d_work = float('%.2f' % (self.t_wait / self.t_work))
        # Calculate occupied nurses
        for o in range(len(self.nl)):
            self.wait_s.append(self.nl[o].ws)
            self.wait_sl.append("Nurse " + str(self.nl[o].l) + " Skill " + str(self.nl[o].s))
            if len(self.nl[o].r) > 2:
                self.on += 1
        if self.on != 0:
            self.a_wait_j = float('%.2f' % (self.t_wait / self.fd_sum))
            self.a_wait_n = float('%.2f' % (self.t_wait / self.on))
            self.a_work_j = float('%.2f' % (self.t_work / self.fd_sum))
            self.a_work_n = float('%.2f' % (self.t_work / self.on))

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
        self.info_to_excel = copy.deepcopy([indexes, skill_level, workload, waiting, avg_waiting, route_e, route_j,
                                            fulfilled_d, self.rewards, self.wait_s])

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
                                    'Fulfilled demands': self.info_to_excel[7],
                                    'Reward': self.info_to_excel[8],
                                    'Waiting time': self.info_to_excel[9]})
        solution_df.to_excel(writer, sheet_name='Sheet1', index=False)
        writer.save()

    def create_fig_text(self, running_time):
        if running_time != 0:
            self.fig_t = "\n" \
                         "Training time: " + str(float('%.2f' % running_time)) + " mins \n" \
                         "Results of best solution found:\n" \
                         "Solution total reward = " + str(self.ev) + "\n" \
                         "Solution occupied nurses = " + str(self.on) + "\n" \
                         "Solution avg workload of nurse = " + str(self.a_work_n) + "\n" \
                         "Solution avg waiting time of job = " + str(self.a_wait_j) + "\n" \
                         "Solution avg waiting time of nurse = " + str(self.a_wait_n) + "\n" \
                         "Solution waiting time proportion = " + str(self.wait_d_work) + "\n" \
                         "Solution remaining demands = " + str(self.rd)
        else:
            self.fig_t = "\n" \
                         "Results of testing the trained agent:\n" \
                         "Solution total reward = " + str(self.ev) + "\n" \
                         "Solution occupied nurses = " + str(self.on) + "\n" \
                         "Solution avg workload of nurse = " + str(self.a_work_n) + "\n" \
                         "Solution avg waiting time of job = " + str(self.a_wait_j) + "\n" \
                         "Solution avg waiting time of nurse = " + str(self.a_wait_n) + "\n" \
                         "Solution waiting time proportion = " + str(self.wait_d_work) + "\n" \
                         "Solution remaining demands = " + str(self.rd)


if __name__ == '__main__':

    """-----Instance representations-----"""
    # Get elder data and distance matrix
    # File index = A,  B,  C,   D
    # File scale = 61, 61, 160, 304
    file_index = 'D'
    file_marks = 'D1'
    file_scale = 304
    e_jobs = []
    e_init_distance_matrix = QL_BWACO.get_data(file_index, file_scale, e_jobs)
    # Read random preference matrix from file
    e_random_pre = np.loadtxt("./initial_preference_D1.csv", delimiter=",", skiprows=0)
    # Skill demand matching parameter matrix
    e_service_time_mean = np.array([[0, 0, 0, 0],
                                    [0, 25, -1, -1],
                                    [0, 20, 30, -1],
                                    [0, 18, 20, 20]])
    e_walk_speed = 60

    """-----Solution Recording-----"""
    solution_final = Solution()
    axis_sub_evaluation_value_iter = []
    axis_avg_wait_nurse_iter = []
    axis_evaluation_value_iter = []
    axis_total_reward_iter = []
    axis_avg_q_matrix_iter = []

    """-----Q Learning parameters-----"""
    ql_learning_rate_para = 0.05
    ql_discount_para = 0.95
    ql_greedy_para = 0.3
    ql_q_matrix_new = np.zeros((20, 3))
    q_matrix_name = "Q matrix - " + file_marks + ".csv"
    # Read q matrix from file
    # ql_q_matrix_new = np.loadtxt("./" + q_matrix_name, delimiter=",", skiprows=0)

    """-----ACO parameters-----"""
    aco_alpha = 1
    aco_beta = 5
    aco_rho = 0.4
    aco_ant_number = 20
    aco_pheromone = 10
    aco_increment = 10
    aco_pheromone_matrix = np.ones((len(e_init_distance_matrix[0]), len(e_init_distance_matrix[0]))) * aco_pheromone

    """-----CCP model parameters-----"""
    ccp_acquaintance_increment = 0.05
    ccp_alpha = 1.29
    ccp_beta = 1.29
    ccp_waiting_limit = 40
    ccp_workload_limit = 480

    e_nurses_skill = [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3]
    # Random or constant preference matrix
    e_preference_ran_matrix = np.row_stack((np.zeros((1, len(e_nurses_skill))), e_random_pre))
    e_preference_one_matrix = np.ones((len(e_init_distance_matrix[0]), len(e_nurses_skill)))

    """-----Start iteration-----"""
    # Record the starting time of the training process
    start = time.time()
    # Data structure: state[0, 0, 0, 0, 0, 0] + action[0] + q value[0]
    training_data = [[[0, 0, 0, 0, 0, 0, 0]], [0]]
    iter_max = 1000
    print('\nCurrent Experimental Instance is ' + file_index)
    print('Instance Scale: ' + str(file_scale-1))
    print('Initial greedy rate: ' + str(ql_greedy_para))
    print('Start training...')
    for iter in range(1000):

        # Solution objective for current sub-solution
        sub_solution = Solution()
        # Changeable list of targets
        available_targets = copy.deepcopy(e_jobs)
        # Delete depot
        available_targets.remove(available_targets[0])
        # Changeable preference, pheromone matrix and nurse's skill set
        changeable_preference_matrix = copy.deepcopy(e_preference_ran_matrix)
        changeable_pheromone_matrix = copy.deepcopy(aco_pheromone_matrix)
        changeable_nurses_skill = copy.deepcopy(e_nurses_skill)
        # Start Q Learning process
        ql_result = QL_BWACO.q_learning(sub_solution.nl, available_targets, changeable_nurses_skill,
                                        ql_q_matrix_new, ql_learning_rate_para, ql_discount_para, ql_greedy_para,
                                        # ACO variables
                                        ccp_acquaintance_increment, ccp_alpha, ccp_beta, ccp_waiting_limit,
                                        ccp_workload_limit,
                                        aco_alpha, aco_beta, aco_rho, aco_ant_number, changeable_pheromone_matrix, aco_increment,
                                        e_walk_speed, changeable_preference_matrix, e_init_distance_matrix,
                                        e_service_time_mean,
                                        e_jobs[0])
        # Record training data
        training_data[0].extend(ql_result[4])
        training_data[1].extend(ql_result[5])
        # Calculate fulfilled and remaining demands
        sub_solution.calculate(ql_result, available_targets)
        # Update global solution according to evaluation value, which is total reward gained in the current episode
        if solution_final.ev < sub_solution.ev:
            solution_final = copy.deepcopy(sub_solution)

        if iter == 100:
            ql_greedy_para = 0.7
            print('Reset greedy rate to: ' + str(ql_greedy_para))

        if iter == 400:
            ql_greedy_para = 0.99
            print('Reset greedy rate to: ' + str(ql_greedy_para))

        axis_avg_wait_nurse_iter.append(solution_final.a_wait_n)
        axis_evaluation_value_iter.append(solution_final.ev)
        axis_sub_evaluation_value_iter.append(sub_solution.ev)
        axis_avg_q_matrix_iter.append(float('%.2f' % ql_q_matrix_new.mean()))

    # Record the ending time of the training process
    end = time.time()
    rt = (end - start) / 60
    # Create figure text for the solution
    solution_final.create_fig_text(rt)
    print(solution_final.fig_t)
    # print("Final greedy rate: " + str(ql_greedy_para))

    # Save training data
    td = "training_data - " + file_marks + ".xls"
    exc = xlwt.Workbook()
    exc.add_sheet("Sheet1")
    exc.save(td)
    writer = pd.ExcelWriter(td, sheet_name='Sheet1')
    solution_df = pd.DataFrame({'Inputs': training_data[0],
                                'Labels': training_data[1]})
    solution_df.to_excel(writer, sheet_name='Sheet1', index=False)
    writer.save()

    # Save detailed information into an excel file
    solution_final.get_solution_info()
    fs = "final solution - " + file_marks + ".xls"
    solution_final.save_solution_info(fs)

    # Plot figures
    iev = "iter - the best total reward - " + file_marks
    Plot.plot(iev, "the best total reward", "iteration", iter_max - 1, axis_evaluation_value_iter, solution_final.fig_t)

    isev = "iter - total reward - " + file_marks
    Plot.plot(isev, "total reward", "iteration", iter_max - 1, axis_sub_evaluation_value_iter, solution_final.fig_t)

    iawt = "iter - avg waiting time of nurse - " + file_marks
    Plot.plot(iawt, "avg waiting time", "iteration", iter_max - 1, axis_avg_wait_nurse_iter, solution_final.fig_t)

    iaqm = "iter - avg q matrix - " + file_marks
    Plot.plot(iaqm, "avg q matrix", "iteration", iter_max - 1, axis_avg_q_matrix_iter, solution_final.fig_t)

    nws = "Nurse waiting sequences - " + file_marks
    Plot.plots(nws, solution_final.wait_sl, solution_final.wait_s, ccp_waiting_limit)

    Plot.heat_map_plot(ql_q_matrix_new, "Q matrix - " + file_marks)

    gf = "Gantt - " + file_marks + " - nurse "
    for item in solution_final.nl:
        if len(item.aT) != 0:
            Plot.gantt(item.aT, gf + str(item.l))

    # Show the trained Q matrix on the console
    print("\nQ matrix:")
    print(np.around(ql_q_matrix_new, decimals=2))
    np.savetxt(q_matrix_name, ql_q_matrix_new, delimiter=',')

    """Solution after training"""
    print("\nStart testing...")
    for ex in range(10):
        print("Test: " + str(ex))
        test_solution = Solution()
        test_targets = copy.deepcopy(e_jobs)
        test_targets.remove(test_targets[0])
        test_nurses = copy.deepcopy(e_nurses_skill)
        trained_q_matrix = copy.deepcopy(ql_q_matrix_new)
        test_pheromone_matrix = copy.deepcopy(aco_pheromone_matrix)
        test_preference_matrix = copy.deepcopy(e_preference_one_matrix)
        test_ql_results = QL_BWACO.q_learning_test(test_solution.nl, test_targets, test_nurses,
                                                   trained_q_matrix, 1,
                                                   # ACO para
                                                   ccp_acquaintance_increment, ccp_alpha, ccp_beta,
                                                   ccp_waiting_limit,
                                                   ccp_workload_limit,
                                                   aco_alpha, aco_beta, aco_rho, aco_ant_number,
                                                   test_pheromone_matrix, aco_increment,
                                                   e_walk_speed, test_preference_matrix, e_init_distance_matrix,
                                                   e_service_time_mean, e_jobs[0])
        test_solution.calculate(test_ql_results, test_targets)
        test_solution.create_fig_text(0)
        print(test_solution.fig_t)
        test_solution.get_solution_info()
        ts = "test solution - " + file_marks + " - " + str(ex) + ".xls"
        test_solution.save_solution_info(ts)
