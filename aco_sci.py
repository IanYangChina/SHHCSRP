# -*- coding: utf-8 -*-
import os

os.getcwd()
import numpy as np
import copy
import pandas as pd
import math
import random
import time
import matplotlib.pyplot as plt
import xlwt

"""
import pandas as pd
import xlrd

读取时间窗表格
tw = pd.read_excel('./TimeWindows.xlsx', sheet_name='Sheet1')
for k in range(int(timeWindows.shape[0]-1)):    # 52
    a = tw['A'][k]
    b = tw['B'][k]
    a = float('%.2f' % a)
    b = float('%.2f' % b)
    timeWindows[k+1].append(a)
    timeWindows[k+1].append(b)
print(timeWindows)
"""

#护工类
class Nurses:
    def __init__(self, label, skill, totalTime, waitingTime, totalDistance):
        self.l = label
        self.s = skill      # Skills = 1, 2, 3
        self.tt = totalTime
        self.wt = waitingTime
        self.td = totalDistance
        self.r = []     # visited elders sequence, i.e., route
        self.sd1 = 0
        self.sd2 = 0
        self.sd3 = 0
        self.aT = []

    def printInfo(self):
        print('Nurse No. ' + str(self.l))
        print('skill level: ' + str(self.s))
        print('total working time: ' + str(self.tt))
        print('total waiting time: ' + str(self.wt))
        print('total distance: ' + str(self.td))
        print('visiting sequence (route): '+ str(self.r))
#老人类
class Elders:
    def __init__(self, lable, requiredJobNum, coordinateX, coordinateY, timeWindowBegin, timeWindowEnd):
        self.l = lable
        self.rj = requiredJobNum         # Jobs = 0, 1, 2, 3
        self.c = [coordinateX, coordinateY]
        self.twb = timeWindowBegin
        self.twe = timeWindowEnd

    def printInfo(self):
        print(self.l, self.rj, self.c, self.twb, self.twe)

class Jobs:
    def __init__(self, lable, elder, level, coordinateX, coordinateY, timeWindowBegin, timeWindowEnd):
        self.l = lable
        self.e = elder
        self.lv = level  # Jobs = 0, 1, 2, 3
        self.c = [coordinateX, coordinateY]
        self.twb = timeWindowBegin
        self.twe = timeWindowEnd
    def printInfo(self):
        print(self.l, self.e, self.lv, self.c, self.twb, self.twe)

def getDistanceMatrix(eldersLocations):
    num = len(eldersLocations)  # 坐标矩阵行数
    distanceMatrix = np.zeros((num, num))
    for i in range(num):
        for j in range(num):
            hD = math.sqrt(pow((eldersLocations[i][1]-eldersLocations[j][1]), 2) + pow((eldersLocations[i][2]-eldersLocations[j][2]), 2))
            if hD == 0:
                distanceMatrix[i][j] = distanceMatrix[j][i] = 9.6 * abs(eldersLocations[i][3] - eldersLocations[j][3])
            else:
                distanceMatrix[i][j] = distanceMatrix[j][i] = hD + eldersLocations[i][3] + eldersLocations[j][3]
    return distanceMatrix

def getElderExcelData(file, eldersIndexes, eldersLocations, jobNum, jobLevels, jobCoordinates, timeWindows):
    excel = pd.read_excel(file, sheet_name='Sheet1')
    jobNum.append(0)
    jobNum += (list(copy.deepcopy(excel['JobNum'].values)))
    eldersIndexes.append(0)
    eldersIndexes += (list(copy.deepcopy(excel['Indexes'].values)))
    jobLevels.append(0)
    jobLevels += (list(copy.deepcopy(excel['JobLevel'].values)))

    jobCoordinates[0][0] = 125
    jobCoordinates[0][1] = 125
    jobCoordinates[0][2] = 0
    timeWindows[0][0] = 0.00
    timeWindows[0][1] = 480.00

    xy = np.vstack((copy.deepcopy(excel['X'].values), copy.deepcopy(excel['Y'].values), copy.deepcopy(excel['Z'].values)))
    for i in range(len(xy[0])):
        jobCoordinates[i+1][0] = xy[0][i]
        jobCoordinates[i+1][1] = xy[1][i]
        jobCoordinates[i+1][2] = xy[2][i]
    tw = np.vstack((copy.deepcopy(excel['TWB'].values), copy.deepcopy(excel['TWE'].values)))
    for i in range(len(tw[0])):
        timeWindows[i+1][0] = tw[0][i]
        timeWindows[i+1][1] = tw[1][i]
    lo = []
    for i in range(fileScale):
        lo.append([eldersIndexes[i], jobCoordinates[i][0], jobCoordinates[i][1], jobCoordinates[i][2]])
    for i in range(fileScale):
        if lo[i] in eldersLocations:
            continue
        else:
            eldersLocations.append(lo[i])

def arrivalTimeCluster(waitingC, currentTime, currentJobs, distanceMatrix, visitingList, perfectTargets):
    for j in range(len(visitingList)):
        distance = distanceMatrix[currentJobs.e][visitingList[j].e]  # 计算距离
        travelTime = distance / walkSpeed  # 计算移动耗时
        arrivalTime = currentTime + travelTime  # 计算到达时刻
        if arrivalTime < visitingList[j].twe:  # 到达不晚于时间窗下限
            if (arrivalTime + waitingC)  >= visitingList[j].twb:  # 到达是否在时间窗内
                perfectTargets.append(visitingList[j])
                continue
        else:
            continue

def getDemands(availableJobs):
    class J:
        def __init__(self, index, demand):
            self.i = index
            self.d = demand
    job = [J(1,0), J(2, 0), J(3, 0)]
    for j in range(len(availableJobs)):
        if availableJobs[j].lv == 1:
            job[0].d += 1
        if availableJobs[j].lv == 2:
            job[1].d += 1
        if availableJobs[j].lv == 3:
            job[2].d += 1
    return([job[0].d, job[1].d, job[2].d])

def stateIdentify(availableJobs,nursesSkill):
    a = len(nursesSkill)
    if a != 0:
        d1 = 0
        d2 = 0
        d3 = 0
        for j in range(len(availableJobs)):
            if availableJobs[j].lv == 1:
                d1 += 1
            if availableJobs[j].lv == 2:
                d2 += 1
            if availableJobs[j].lv == 3:
                d3 += 1
        d = d1 + d2 + d3
        if d != 0:
            if d1 > d2:
                if d2 > d3:
                    return(0)   # 0-d1>d2>d3
                else:
                    if d1 > d3:
                        return(1)   # 1-d1>d3>d2
                    else:
                        return(4)   # 4-d3>d1>d2
            else:   # d2>d1
                if d1 > d3:
                    return(2)   # 2-d2>d1>d3
                else:
                    if d3 > d2:
                        return(5)   # 5-d3>d2>d1
                    else:
                        return(3)   # 3-d2>d3>d1
        else:
            return(6)   # 6-no more demands
    else:
        return(7)   # 7-no more nurses

def actionTaking(availableJobs, nursesSkill):
    a = len(nursesSkill)
    if a != 0:
        d1 = 0
        d2 = 0
        d3 = 0
        for j in range(len(availableJobs)):
            if availableJobs[j].lv == 1:
                d1 += 1
            if availableJobs[j].lv == 2:
                d2 += 1
            if availableJobs[j].lv == 3:
                d3 += 1
        d = d1 + d2 + d3
        if d != 0:
            if d1 > d2:
                if d2 > d3:
                    return(1)
            else:
                if d2 > d3:
                    return(2)
                else:
                    return(3)
        else:
            return(4)
    else:
        return(5)

def qActionTaking(iter, currentState, currentQMatrix, nursesSkill, greedy):
    class Q:
        def __init__(self, qValue, action):
            self.q = qValue
            self.a = action
    g = random.uniform(0, 1)  # e-greedy exploration
    if g < greedy:
        qValues = [Q(currentQMatrix[currentState][0], 1),
                   Q(currentQMatrix[currentState][1], 2),
                   Q(currentQMatrix[currentState][2], 3)]
        qV = sorted(qValues, key=lambda Q: Q.q)
        if qV[2].a in nursesSkill:
            nursesSkill.remove(qV[2].a)
            return(qV[2].a)
        elif qV[1].a in nursesSkill:
            nursesSkill.remove(qV[1].a)
            return(qV[1].a)
        else:
            nursesSkill.remove(qV[0].a)
            return(qV[0].a)
    else:
        skill = copy.deepcopy(nursesSkill)
        random.shuffle(skill)
        action = skill[0]
        nursesSkill.remove(action)
        return (action)

def getSolutionInfo(optimalSolution, distanceMatrix, nursesI, nursesS, nursesWT, nursesR,nursesD,nursesE1, nursesE2, nursesE3, nursesWait):
    for i in range(len(optimalSolution)):
        R = []
        nursesI.append(optimalSolution[i].l)
        nursesS.append(optimalSolution[i].s)
        nursesWT.append(optimalSolution[i].tt)
        nursesWait.append(optimalSolution[i].wt)
        for e in range(len(optimalSolution[i].r)-1):
            optimalSolution[i].td += distanceMatrix[optimalSolution[i].r[e].e][optimalSolution[i].r[e+1].e]/60
        for r in range(len(optimalSolution[i].r)):
            R.append(optimalSolution[i].r[r].e)
        nursesR.append(R)
        nursesD.append(round(optimalSolution[i].td, 2))
        d1 = 0
        d2 = 0
        d3 = 0
        for j in range(len(optimalSolution[i].r)):
            if optimalSolution[i].r[j].lv == 1:
                d1 += 1
            if optimalSolution[i].r[j].lv == 2:
                d2 += 1
            if optimalSolution[i].r[j].lv == 3:
                d3 += 1
        nursesE1.append(d1)
        nursesE2.append(d2)
        nursesE3.append(d3)

def getTimeWindows(nurses, visitedTWI, visitedTW):
    for twi in range(len(nurses.r)):
        visitedTWI.append([twi, twi])
        visitedTW.append([nurses.r[twi].twb, nurses.r[twi].twe])

# learningRate discount greedy workTimePara
ParaQL = [[0.1, 0.9, 0.5, -0.01],   # learning rate 0.9
          [0.5, 0.9, 0.5, -0.01],   # learning rate 0.1
          [0.9, 0.9, 0.5, -0.01]]   # learning rate 0.5

# initialPheromone alpha beta evaporation
ParaBEACO = [[20, 1, 1, 0.1],
             [20, 1, 1, 0.5],
             [20, 5, 1, 0.1],
             [20, 5, 1, 0.5],
             [20, 1, 5, 0.1],
             [20, 1, 5, 0.5]]

randomPre = np.loadtxt("./initial_preference_ABC.csv",delimiter=",",skiprows=0)

# initial perference, decrement, C, workload, a, b

# confidence levels:        100% 95%   90%  80%  70%  60%  50%
# inverse standard normal:  3.09 1.65, 1.29 0.85 0.52 0.26 0
ParaModel = [[1, 0.2, 60, 480, 1.29, 1.29]]

initPheromone = 20
alpha = 1
beta = 1
rho = 0.5  # evaporation

learningRate = 0.9
discount = 0.9
greedy = 0.5
workTimePara = -0.01

c = 1
waitingC = 60
totalWork = 480
alphaModel = 1.29
betaModel = 1.29

"""------数学模型表述------"""
# 读取案例数据
fileIndex = 'A'
fileScale = 0
file = ''
if fileIndex == 'A':
    fileScale = 61
    file = './Elders_' + fileIndex + '.xlsx'
    print('Current Experimental Instance is ' + fileIndex)
    print('Instance Scale: ' + str(fileScale))
eldersIndexes = []
jobNum = []
jobLevels = []
eldersLocations = []
jobCoordinates = np.zeros((fileScale, 3))
timeWindows = np.zeros((fileScale, 2))
getElderExcelData(file, eldersIndexes, eldersLocations, jobNum, jobLevels, jobCoordinates,timeWindows)

# 案例人数
initNursesSkill = [1, 2, 2, 2, 3, 3, 3]
nursesNum = len(initNursesSkill)
eldersNum = len(eldersLocations)
#矩阵
zeros = np.zeros((1,nursesNum))
preference = np.ones((eldersNum-1, nursesNum))  # 相识度
initPreferenceMatch = np.row_stack((zeros, randomPre))

distanceMatrix = getDistanceMatrix(eldersLocations)  # 距离
skillsMatch = np.array([[0, 0, 0, 0],
                        [0, 25, -1, -1],
                        [0, 20, 30, -1],
                        [0, 18, 20, 20]])   # 服务时长参数（根据不同技能-工作组合）
# jobs, level, coordinateX, coordinateY, timeWindowBegin, timeWindowEnd
jobs = []
for fs in range(fileScale):
    jobs.append(
        Jobs(fs, eldersIndexes[fs], jobLevels[fs], jobCoordinates[fs][0], jobCoordinates[fs][1], timeWindows[fs][0],
             timeWindows[fs][1]))
walkSpeed = 60  # 护工移动速度: 40m/min
elevatorSpeed = 105 # 电梯速度： 105m/min
stairsSpeed = 48    # 爬楼梯速度： 48m/min
"""------BWACO Initialization------"""
antNum = 60  # 蚂蚁个数
pheromoneTable = np.ones((eldersNum, eldersNum))  # 信息素矩阵
pheromoneTable *= initPheromone
"""--Q-learning Initialization--"""
currentQMatrix = np.zeros((8,3))    # initialize Q-matrix
state = [0, 1, 2, 3, 4, 5, 6, 7]
# states: 0-d1>d2>d3; 1-d1>d3>d2; 2-d2>d1>d3; 3-d2>d3>d1; 4-d3>d1>d2; 5-d3>d2>d1; 6-d1=d2=d3=0; 7-a1=a2=a3=0
"""---Solutions Recording Variables---"""
optimalTimeAxi = []
subOptimalTimeAxi = []
optimalSolution = []
remainedDemands = []
solutionPrefer = 0
remainedDemandsLevels = []
optimalTime = 10000
start = time.clock()    # 记录迭代起始时刻
iter = 0
itermax = 200

while iter < itermax:
    iter += 1
    availableJobs = copy.deepcopy(jobs) # 初始化可拜访目标
    availableJobs.remove(availableJobs[0])  # 从可拜访集中，删除depot
    preferenceMatch = copy.deepcopy(initPreferenceMatch)    # reset preference matrix per episode
    currentPheromone = copy.deepcopy(pheromoneTable)  # 重置信息素矩阵
    subOptimalTime = 0
    nursesList = []
    visitedNum = 0
    nursesSkill = copy.deepcopy(initNursesSkill)

    for n in range(nursesNum):  # 对每个护工构建路线
        # get current state
        currentState = stateIdentify(availableJobs, nursesSkill)
        if currentState in [6, 7]:
            break   # reach absorbing state
        # take action according to the e-greedy
        currentSkill = qActionTaking(iter, currentState, currentQMatrix, nursesSkill, greedy)
        """
        currentSkill = actionTaking(availableJobs, nursesSkill)
        if currentSkill in [4, 5]:
            break
        """
        # build nurses sequence
        nursesList.append(Nurses(n, currentSkill, 0, 0, 0))
        currentDemands = getDemands(availableJobs)

        s = copy.deepcopy(nursesList[n].s) # 确认当前护工技能水平

        forVisit = []
        for aj in range(len(availableJobs)):    # 从可拜访集中，删除当前护工技能无法满足的目标
            if availableJobs[aj].lv <= s:
                forVisit.append(availableJobs[aj])

        bestPath = []  # 记录当前最优路径
        worstPath = []  # 记录当前最差解
        optimalArrivalTime = []
        shortestTime = 0    # 记录当前最优路径耗时
        waitingTime = 0
        PPM_BestObjective = 0   # record current PPM objective
        PPM_WorstObjective = 0
        for ant in range(antNum): # 对每只ant构建路线

            currentJob = jobs[0]  # 起点初始化为depot
            currentTime = 0  # 起始时间初始化为0
            currentWaiting = 0
            subArrivalTime = []
            offDutyTime = 0  # 下班时间初始化为0
            antPathTable = []  # 路径记录表
            antPathTable.append(jobs[0])

            # 初始化待选集
            visitingList = copy.deepcopy(forVisit)
            currentPreference = copy.deepcopy(preferenceMatch)

            while offDutyTime <= totalWork:   # 下班时间不超时则继续，机会约束（22）

                serviceTimeMean = skillsMatch[s][currentJob.lv]  # 确定当前组合服务时长均值
                preferenceFactor = copy.deepcopy(currentPreference[currentJob.e][nursesList[n].l])
                if currentTime < currentJob.twb:
                    currentWaiting += (currentJob.twb-currentTime)
                    currentTime = copy.deepcopy(currentJob.twb)
                    subArrivalTime.append(copy.deepcopy(currentTime))
                else:
                    subArrivalTime.append(copy.deepcopy(currentTime))
                # 计算当前位置回到depot的时刻
                offDutyTime = currentTime + (preferenceFactor*serviceTimeMean) + betaModel + (distanceMatrix[currentJob.e][jobs[0].e])/walkSpeed
                if offDutyTime >= totalWork:   # 如果护工已经超时则停止路线构建
                    # print('off duty time outs: ' + str(offDutyTime))
                    currentTime += (preferenceFactor*serviceTimeMean) + betaModel + (distanceMatrix[currentJob.e][jobs[0].e])/walkSpeed
                    antPathTable.append(jobs[0]) # 回到depot，停止路径构建
                    subArrivalTime.append(copy.deepcopy(currentTime))
                    break
                # add up service time, Eq.(22),(30)
                currentTime += (preferenceFactor*serviceTimeMean) + alphaModel

                pr = [] # 转移概率
                perfectTargets = [] # 完美到达列表

                arrivalTimeCluster(waitingC, currentTime,currentJob,distanceMatrix, visitingList, perfectTargets)

                # 确定满足约束的最优目标
                if (len(perfectTargets)): # 是否有目标可到达于时间窗之内
                    pD = 0
                    for pdd in range(len(perfectTargets)):
                        waitD = (distanceMatrix[currentJob.e][perfectTargets[pdd].e] / walkSpeed) + currentTime - perfectTargets[pdd].twb
                        pD += (1/waitD)
                    for pt in range(len(perfectTargets)):
                        wait = (distanceMatrix[currentJob.e][perfectTargets[pt].e]/walkSpeed) + currentTime - perfectTargets[pt].twb
                        yita = 1/wait
                        pU = pow((currentPheromone[currentJob.e][perfectTargets[pt].e]), alpha) * pow(yita, beta)
                        pT = pU / pD
                        pr.append([currentJob, perfectTargets[pt], pT])
                else:
                    # print('no targets feasible')    # 无满足要求目标(currentTime >= 480)
                    currentTime += (distanceMatrix[currentJob.e][jobs[0].e])/walkSpeed   # 更新返回depot的时刻
                    currentJob = jobs[0]    # 更新当前位置，回到depot
                    subArrivalTime.append(copy.deepcopy(currentTime))   # record arrival time back to depot
                    antPathTable.append(jobs[0])  # 更新路径表
                    break   # 停止构建路径,跳出while循环

                maxPr = pr[0][2]
                minPr = pr[0][2]
                bestTarget = pr[0][1]
                # 获取最优移动目标
                if len(pr): # 存在满足时间窗的目标，取最大转移概率者
                    for i in range(len(pr)):
                        if maxPr < pr[i][2]:
                            maxPr = pr[i][2]
                            bestTarget = pr[i][1]
                    antPathTable.append(bestTarget)  # 更新路径表
                    currentPreference[bestTarget.e][n] -= c  # 见过面的组合减陌生度
                    currentTime += (distanceMatrix[currentJob.e][bestTarget.e]) / walkSpeed
                    currentJob = bestTarget

                    for v in range(len(visitingList)):
                        if visitingList[v].l == bestTarget.l:
                            visitingList.remove(visitingList[v])
                            break

            fulfilledDemandsAnt = copy.deepcopy(len(antPathTable)-2)
            if fulfilledDemandsAnt == 0:    # no demands fulfilled
                if len(bestPath) == 0:
                    bestPath = copy.deepcopy(antPathTable)
                    worstPath = copy.deepcopy(antPathTable)
            else:
                PPM_Objective = copy.deepcopy(currentTime / fulfilledDemandsAnt)
                if ant == 0:   # 判断是否首次循环
                    PPM_BestObjective = copy.deepcopy(PPM_Objective)    # first iteration, record best PPM objective
                    PPM_WorstObjective = copy.deepcopy(PPM_Objective)   # first iteration, record worst PPM objective
                    shortestTime = copy.deepcopy(currentTime)   # first iteration, record working time
                    waitingTime = copy.deepcopy(currentWaiting) # first iteration, record waiting time
                    bestPath = copy.deepcopy(antPathTable)  # first iteration, record best route
                    worstPath = copy.deepcopy(antPathTable) # first iteration, record worst route
                    optimalArrivalTime = copy.deepcopy(subArrivalTime)
                else:   # 非首次循环
                    if PPM_BestObjective > PPM_Objective:  # 判断新路是否优于前代
                        PPM_BestObjective = copy.deepcopy(PPM_Objective)    # 更新最优值
                        shortestTime = copy.deepcopy(currentTime)
                        waitingTime = copy.deepcopy(currentWaiting)
                        bestPath = copy.deepcopy(antPathTable) # 更新最优路径
                        optimalArrivalTime = copy.deepcopy(subArrivalTime)
                    elif PPM_WorstObjective < currentTime: # 判断新路是否更差
                        PPM_WorstObjective = copy.deepcopy(PPM_Objective)
                        worstPath = copy.deepcopy(antPathTable)
                    else:
                        continue

        # update route
        nursesList[n].tt = copy.deepcopy(shortestTime)
        nursesList[n].aT = copy.deepcopy(optimalArrivalTime)
        nursesList[n].wt = copy.deepcopy(waitingTime)
        for o in range(len(bestPath)):
            nursesList[n].r.append(bestPath[o])
            if bestPath[o].lv == 1:
                nursesList[n].sd1 += 1
            elif bestPath[o].lv == 2:
                nursesList[n].sd2 += 1
            elif bestPath[o].lv == 3:
                nursesList[n].sd3 += 1
            if bestPath[o].l != 0:
                preferenceMatch[bestPath[o].e][nursesList[n].l] -= c
                for b in range(len(availableJobs)):
                    if availableJobs[b].l == bestPath[o].l:
                        availableJobs.remove(availableJobs[b])
                        break

        # updat pheromone according to Best-Worst rule
        for bP in range(len(bestPath)):
            if (bP + 1) == len(bestPath):
                break
            currentPheromone[bestPath[bP].e][bestPath[bP+1].e] = (1 - rho)*currentPheromone[bestPath[bP].e][bestPath[bP+1].e] + distanceMatrix[bestPath[bP].e][bestPath[bP+1].e]
        for wP in range(len(worstPath)):
            if (wP + 1) == len(worstPath):
                break
            currentPheromone[worstPath[wP].e][worstPath[wP+1].e] = (1 - rho)*currentPheromone[worstPath[wP].e][worstPath[wP+1].e] - distanceMatrix[worstPath[wP].e][worstPath[wP+1].e]

        # update Q-matrix
        demands = getDemands(availableJobs)
        # calculate rewards
        rDW = (currentDemands[0] - demands[0]) + (currentDemands[1] - demands[1]) + (currentDemands[2] - demands[2]) + workTimePara*shortestTime
        qValue = (1-learningRate)*currentQMatrix[currentState][currentSkill-1] + learningRate*(rDW + discount*currentQMatrix[currentState][currentSkill-1])
        currentQMatrix[currentState][currentSkill-1] = float('%.2f' % (copy.deepcopy(qValue)))

        # record sub solutions
        t = float('%.2f' % (copy.deepcopy(nursesList[n].tt)))
        subOptimalTime += t

    if subOptimalTime < optimalTime:
        # update better solutions
        optimalTime = float('%.2f' % (copy.deepcopy(subOptimalTime)))
        optimalSolution = copy.deepcopy(nursesList)
        remainedDemands = copy.deepcopy(getDemands(availableJobs))
        solutionPrefer = copy.deepcopy(preferenceMatch)
        rd = []
        for d in range(len(availableJobs)):
            rd.append(copy.deepcopy([availableJobs[d].l, availableJobs[d].lv]))
        remainedDemandsLevels = copy.deepcopy(rd)

    subOptimalTimeAxi.append(float('%.2f' % (copy.deepcopy(subOptimalTime))))
    optimalTimeAxi.append(optimalTime)

    # output solution information
    if iter == itermax:
        end = time.clock()
        cpu = float('%.2f' % (copy.deepcopy(end - start)))

        lr = 'Learning rate: ' + str(learningRate)
        gf = 'Greedy factor: ' + str(greedy)
        df = 'Discount factor: ' + str(discount)
        wf = 'Working time factor: ' + str(workTimePara)

        af = 'α: ' + str(alpha)
        bt = 'β: ' + str(beta)
        lou = 'ρ: ' + str(rho)
        initPher = "τo: " + str(initPheromone)

        # confidence levels:        100% 95%   90%  80%  70%  60%  50%
        # inverse standard normal:  3.09 1.65, 1.29 0.85 0.52 0.26 0
        A = 0
        B = 0
        if alphaModel == 3.09:
            A = 1
        elif alphaModel == 1.65:
            A = 0.95
        elif alphaModel == 1.29:
            A = 0.9
        elif alphaModel == 0.85:
            A = 0.8
        elif alphaModel == 0.52:
            A = 0.7
        elif alphaModel == 0.26:
            A = 0.6
        elif alphaModel == 0:
            A = 0.5

        if betaModel == 3.09:
            B = 1
        elif betaModel == 1.65:
            B = 0.95
        elif betaModel == 1.29:
            B = 0.9
        elif betaModel == 0.85:
            B = 0.8
        elif betaModel == 0.52:
            B = 0.7
        elif betaModel == 0.26:
            B = 0.6
        elif betaModel == 0:
            B = 0.5

        preferIncre = 'c: ' + str(c)
        confidence = 'Confidence α, β: ' + str(A) + ', ' + str(B)
        waitingLimit = 'Waiting Bound C: ' + str(waitingC)
        workLoad = 'Total Workload: ' + str(totalWork)
        initPrefer = 'Initial Preference Factor: 1'

        cp = 'CPU time: ' + str(cpu)
        opt = 'Optimal Working Time: ' + str(optimalTime)
        dd = 'Unfulfilled Demands: Lv.1: ' + str(remainedDemands[0]) + ', Lv.2: ' + str(remainedDemands[1]) + ', Lv.3: ' + str(remainedDemands[2])

        print('iteration: ' + str(iter))
        np.around(solutionPrefer, decimals=3)
        np.around(currentQMatrix, decimals=3)
        pQM = 'Pre_I' + str(c) + ' C' + str(waitingC) + ' ToW_' + str(totalWork) + ' A_' + str(A) + ' B_' + str(B) + ' IPre1_solutionQMatrix.csv'
        np.savetxt(pQM, currentQMatrix, delimiter=',')
        pFN = 'Pre_I' + str(c) + ' C' + str(waitingC) + ' ToW_' + str(totalWork) + ' A_' + str(A) + ' B_' + str(B) + ' IPre1_solutionPrefer.csv'
        np.savetxt(pFN, solutionPrefer, delimiter=',')
        print('Solution informetion: ')
        print(cp, opt, dd)

        # save solution information into excel
        nursesI = []
        nursesS = []
        nursesWT = []
        nursesR = []
        nursesD = []
        nursesE1 = []
        nursesE2 = []
        nursesE3 = []
        nursesWait = []
        getSolutionInfo(optimalSolution, distanceMatrix, nursesI,nursesS, nursesWT, nursesR, nursesD, nursesE1,nursesE2, nursesE3, nursesWait)
        exc = xlwt.Workbook()
        eName = 'Pre_I' + str(c) + ' C' + str(waitingC) + ' ToW_' + str(totalWork) + ' A_' + str(A) + ' B_' + str(B) + ' IPre1_solutions.xls'
        sheet = exc.add_sheet("Sheet1")
        exc.save(eName)
        writer = pd.ExcelWriter(eName, sheet_name='Sheet1')
        solution = pd.DataFrame({'Nurses':nursesI,
                                 'Skill':nursesS,
                                 'Working Time':nursesWT,
                                 'Traveling Time':nursesD,
                                 'Waiting Time': nursesWait,
                                 'Routes': nursesR,
                                 'Demands_1': nursesE1,
                                 'Demands_2': nursesE2,
                                 'Demands_3': nursesE3})
        solution.to_excel(writer, sheet_name='Sheet1')
        writer.save()

        # paint optimal time trail
        x = 75
        y = optimalTime + 700
        plt.ylabel('optimal time (min)')
        plt.xlabel('iteration')
        plt.xlim(0, iter)
        plt.ylim(y - 800, y)
        plt.plot(optimalTimeAxi)
        # Plot QL Paras: lr, df, gf, wf
        # plt.text(x, y - 50, 'Parameters', fontsize=9, verticalalignment="top", horizontalalignment="left")
        # plt.text(x, y - 105, lr, fontsize=9, verticalalignment="top",horizontalalignment="left")
        # plt.text(x, y - 160, gf, fontsize=9, verticalalignment="top",horizontalalignment="left")
        # plt.text(x, y - 215, df, fontsize=9, verticalalignment="top", horizontalalignment="left")
        # plt.text(x, y - 270, wf, fontsize=9, verticalalignment="top", horizontalalignment="left")
        # plt.text(x, y - 325, 'Results', fontsize=9, verticalalignment="top", horizontalalignment="left")
        # plt.text(x, y - 380, cp, fontsize=9, verticalalignment="top", horizontalalignment="left")
        # plt.text(x, y - 435, dd, fontsize=9, verticalalignment="top", horizontalalignment="left")
        # plt.text(x, y - 490, opt, fontsize=9, verticalalignment="top", horizontalalignment="left")
        # Plot BWACO Paras: af, bt, lou, initPher
        # plt.text(x, y - 50, 'Parameters', fontsize=9, verticalalignment="top", horizontalalignment="left")
        # plt.text(x, y - 105, af, fontsize=9, verticalalignment="top", horizontalalignment="left")
        # plt.text(x, y - 160, bt, fontsize=9, verticalalignment="top",horizontalalignment="left")
        # plt.text(x, y - 215, lou, fontsize=9, verticalalignment="top", horizontalalignment="left")
        # plt.text(x, y - 270, initPher, fontsize=9, verticalalignment="top", horizontalalignment="left")
        # plt.text(x, y - 325, 'Results', fontsize=9, verticalalignment="top", horizontalalignment="left")
        # plt.text(x, y - 380, cp, fontsize=9, verticalalignment="top", horizontalalignment="left")
        # plt.text(x, y - 435, dd, fontsize=9, verticalalignment="top", horizontalalignment="left")
        # plt.text(x, y - 490, opt, fontsize=9, verticalalignment="top", horizontalalignment="left")
        # Plot Model Paras: preferIncre, workLoad, waitingLimit, initPrefer, confidence
        plt.text(x, y - 50, 'Parameters', fontsize=9, verticalalignment="top", horizontalalignment="left")
        plt.text(x, y - 105, preferIncre, fontsize=9, verticalalignment="top",horizontalalignment="left")
        plt.text(x, y - 160, workLoad, fontsize=9, verticalalignment="top",horizontalalignment="left")
        plt.text(x, y - 215, waitingLimit, fontsize=9, verticalalignment="top", horizontalalignment="left")
        plt.text(x, y - 270, initPrefer, fontsize=9, verticalalignment="top", horizontalalignment="left")
        plt.text(x, y - 325, confidence, fontsize=9, verticalalignment="top", horizontalalignment="left")
        plt.text(x, y - 380, 'Results', fontsize=9, verticalalignment="top", horizontalalignment="left")
        plt.text(x, y - 435, cp, fontsize=9, verticalalignment="top", horizontalalignment="left")
        plt.text(x, y - 490, dd, fontsize=9, verticalalignment="top", horizontalalignment="left")
        plt.text(x, y - 545, opt, fontsize=9, verticalalignment="top", horizontalalignment="left")
        name = 'Pre_I' + str(c) + ' C' + str(waitingC) + ' ToW_' + str(totalWork) + ' A_' + str(A) + ' B_' + str(B) + ' IPre1_optimalTime.png'
        plt.savefig(name, dpi=900, bbox_inches='tight')
        plt.close()

        # paint time windows & arrival time
        for nu in range(len(optimalSolution)):
            if len(optimalSolution[nu].r) < 3:
                continue
            visitedTW = []
            visitedTWI = []
            getTimeWindows(optimalSolution[nu], visitedTWI, visitedTW)

            plt.ylabel('time window (min)')
            plt.xlabel('elders index')
            plt.xlim(-1, len(visitedTWI))
            plt.ylim(-20, 500)
            for tw in range(len(visitedTWI)):
                plt.plot(visitedTWI[tw], visitedTW[tw], color='#0000FF', linestyle='-')
            aTL = copy.deepcopy(optimalSolution[nu].aT)
            for at in range(len(aTL) - 1):
                plt.plot([at, at + 1], [aTL[at], aTL[at + 1]], color='#000000', linestyle=':')
            name = 'Pre_I' + str(c) + ' C' + str(waitingC) + ' ToW_' + str(totalWork) + ' A_' + str(A) + ' B_' + str(B) + ' IPre1_arrivalTimeTrail_' + str(nu)  + '.png'
            ft = float('%.2f' % (copy.deepcopy(aTL[len(aTL) - 1])))
            finalArrival = 'Nurses skill: ' + str(optimalSolution[nu].s) + ' Final arrival time: ' + str(ft)
            plt.title(finalArrival)
            plt.savefig(name, dpi=900, bbox_inches='tight')
            plt.close()

        # Sup-optimal solutions trail
        """
        x = 75
        y = optimalTime + 700
        plt.ylabel('working time (min)')
        plt.xlabel('iteration')
        plt.xlim(0, iter)
        plt.ylim(y - 800, y)
        plt.plot(subOptimalTimeAxi)
        plt.text(x, y - 50, 'Parameters', fontsize=9, verticalalignment="top", horizontalalignment="left")
        plt.text(x, y - 105, lr, fontsize=9, verticalalignment="top", horizontalalignment="left")
        plt.text(x, y - 160, df, fontsize=9, verticalalignment="top", horizontalalignment="left")
        plt.text(x, y - 215, gf, fontsize=9, verticalalignment="top", horizontalalignment="left")
        name =  'QL_G' + str(greedy) + ' L' + str(learningRate) + ' D' + str(discount) + ' R' + str(
            workTimePara) + ' ACO_A' + str(alpha) + ' B' + str(beta) + ' L' + str(rho) + ' IPh' + str(
            initPheromone) + ' Pre_I' + str(c) + ' C' + str(waitingC) + 'IPre1_suboptimalTime.png'
        plt.savefig(name, dpi=900, bbox_inches='tight')
        plt.close()
        """