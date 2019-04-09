import numpy as np
import matplotlib.pyplot as plt


# Plot multiple tendency lines in 1 figure
def plots(figure_name, y_label, y, waiting_limit):
    plt.rcParams['figure.figsize'] = (6.0, 4.0)
    markers = ['o', 'v', '*', '^', 's', 'p', 'h', '+', 'x', 'D', '1', '8', 'P', 'o', 'v', '*', '^', 's', 'p', 'x', 'D']
    x = 0
    for l in range(len(y)):
        if x < len(y[l]):
            x = len(y[l])
    plt.xticks(np.arange(x))
    plt.ylim(-5, waiting_limit + 5)
    plt.xlabel('Sequentially-numbered demands', fontsize=13)
    plt.ylabel('Waiting time', fontsize=13)

    for n in range(len(y)):
        if len(y[n]) != 0:
            plt.plot(y[n], marker=markers[n], markersize=5, label=y_label[n])

    plt.legend(loc='upper right', frameon=False, fontsize=10)
    name = figure_name + '.png'
    plt.savefig(name, dpi=900, bbox_inches='tight')
    plt.close()


# Plot tendency graphs
def plot(figure_name, y_label, x_label, x, y, figure_text):
    plt.rcParams['figure.figsize'] = (6.0, 4.0)
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


# Plot heat-maps for preference and Q matrices
def heat_map_plot(matrix, file_name):
    plt.rcParams['figure.figsize'] = (6.0, 4.0)
    fig, ax = plt.subplots()

    im = ax.imshow(matrix, cmap="YlGn")
    ax.figure.colorbar(im, ax=ax, cmap="YlGn")
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    ax.set_xticklabels([1, 2, 3], fontsize=7)
    ax.set_yticklabels(['d1>=d2>=d3_1', 'd1>=d2>=d3_2', 'd1>=d2>=d3_3+',
                        'd1>=d3>=d2_1', 'd1>=d3>=d2_2', 'd1>=d3>=d2_3+',
                        'd2>=d1>=d3_1', 'd2>=d1>=d3_2', 'd2>=d1>=d3_3+',
                        'd2>=d3>=d1_1', 'd2>=d3>=d1_2', 'd2>=d3>=d1_3+',
                        'd3>=d1>=d2_1', 'd3>=d1>=d2_2', 'd3>=d1>=d2_3+',
                        'd3>=d2>=d1_1', 'd3>=d2>=d1_2', 'd3>=d2>=d1_3+',
                        '0 demands', '0 nurses'], fontsize=7)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    fig.tight_layout()
    plt.savefig(file_name + ".png", dpi=900, bbox_inches='tight')
    plt.close()


# Plot waiting sequences
def subplots(y1, y2, figure_name, xticklabels, linelabels, ax1_ylabel, ax2_ylabel):
    plt.rcParams['figure.figsize'] = (6.0, 4.0)
    markers = ['o', 'v', '*', '^', 's', 'p', 'h', '+', 'x', 'D']
    fig = plt.figure(1)
    ax1 = fig.add_subplot(211)
    ax1.set_xticks(np.arange(7))
    ax1.set_xticklabels(xticklabels, fontsize=8)
    ax1.set_ylabel(ax1_ylabel, fontsize=8)
    ax1.set_xlabel("Confidence level of waiting limitation", fontsize=8)
    for yy1 in range(len(y1)):
        ax1.plot(y1[yy1], marker=markers[yy1], markersize=5, label=linelabels[yy1])
    ax1.legend(loc='upper right', frameon=False, fontsize=8)
    ax1.set_ylim(2.0, 5.0)
    ax1.set_yticks(np.arange(1.0, 5.0, 0.5))
    ax1.set_yticklabels(np.arange(1.0, 5.0, 0.5), fontsize=8)

    ax2 = fig.add_subplot(212)
    ax2.set_xticks(np.arange(7))
    ax2.set_xticklabels(xticklabels, fontsize=8)
    ax2.set_ylabel(ax2_ylabel, fontsize=8)
    ax2.set_xlabel("Confidence level of workload limitation", fontsize=8)
    for yy2 in range(len(y2)):
        ax2.plot(y2[yy2], marker=markers[yy2], markersize=5)
    ax2.set_ylim(390, 420)
    ax2.set_yticks(np.arange(390, 420, 5))
    ax2.set_yticklabels(np.arange(390, 420, 5), fontsize=8)

    """
    ax3.set_title(ax3_label, fontsize=8)
    ax3.set_xticks(np.arange(7))
    ax3.set_xticklabels(xticklabels, fontsize=8)
    ym3 = 0
    for yy3 in range(len(y3)):
        ax3.plot(y3[yy3], marker=markers[yy3], markersize=5)
        if ym3 < max(y3[yy3]):
            ym3 = max(y3[yy3])
    ax3.set_ylim(430, ym3+20)
    ax3.set_yticks(np.arange(430, ym3+20, 10))
    ax3.set_yticklabels(np.arange(430, ym3+20, 10), fontsize=8)
    """
    plt.tight_layout()
    name = figure_name + '.png'
    plt.savefig(name, dpi=900, bbox_inches='tight')
    plt.close()


# Plot Gantt Diagram
def gantt(time_schedule, figure_name):
    plt.rcParams['figure.figsize'] = (16.0, 9.0)
    ax = plt.gca()
    [ax.spines[i].set_visible(False) for i in ["top", "right"]]
    x = []
    y = []
    time_schedule.reverse()
    for i in range(len(time_schedule)):
        if i == 0:
            y.append(time_schedule[i][0])
            x.append(time_schedule[i][1])
            plt.plot([time_schedule[i][1], time_schedule[i][1]], [-1, i + 1.2],
                     color='black', linestyle='--', linewidth=0.4)
            plt.barh(i, time_schedule[i][2], height=0.6, color='r', hatch='///', label='Waiting costs',
                     left=time_schedule[i][1])
            plt.barh(i, time_schedule[i][3], height=0.6, color='b', hatch='++', label='Service time',
                     left=time_schedule[i][1] + time_schedule[i][2])
            plt.barh(i, time_schedule[i][4], height=0.6, color='y', hatch='oo', label='Travel costs',
                     left=time_schedule[i][1] + time_schedule[i][2] + time_schedule[i][3])
        else:
            y.append(time_schedule[i][0])
            x.append(time_schedule[i][1])
            plt.plot([time_schedule[i][1], time_schedule[i][1]], [-1, i + 1.2],
                     color='black', linestyle='--', linewidth=0.4)
            plt.barh(i, time_schedule[i][2], height=0.6, color='r', hatch='///',
                     left=time_schedule[i][1])
            plt.barh(i, time_schedule[i][3], height=0.6, color='b', hatch='++',
                     left=time_schedule[i][1] + time_schedule[i][2])
            plt.barh(i, time_schedule[i][4], height=0.6, color='y', hatch='oo',
                     left=time_schedule[i][1] + time_schedule[i][2] + time_schedule[i][3])
    plt.ylabel('Elders\' labels')
    plt.xlabel('Time line starting from 8 A.M. / min')
    plt.yticks(np.arange(len(time_schedule)), y, size=10)
    plt.xticks(x[:-1], size=10)
    plt.ylim(-0.1, len(time_schedule))
    plt.legend(loc='upper right', frameon=False, fontsize=12)
    name = figure_name + '.png'
    plt.savefig(name, dpi=700)
    plt.close()