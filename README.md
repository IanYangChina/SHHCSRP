# Q-learning---Ant-colony-optimization
The project aims to find a acceptable solution for a multi-nurse, multi-appointment home health care routing problem with stochastic service time and dense locations by Q Learning and Ant Colony Optimisation algorithm. This introduction is separated into three parts, including problem description, algorithm and discussion.

## Problem description.

The problem is formally named Home Health Care Scheduling and Routing Problem (HHCSRP) and has been studied for decades since 1998 (Cheng E et al 1998). It is actually a variant of the well-known Vehicle Routing Problem with some complicated constraints. Basically, the task is to assign a group of demands required by many elders to a group of caregivers with optimised cost. However, mathematical models, their objectives and constraints differ from research to research. As for this project, the model is a hybrid one that includes finite markov decision process (MDPs), chance constrained programming (CCP), and multi-objective optimisation programming (MOOP).

**Objectives:**
1. Maximize the fulfilled demands
2. Minimize the total waiting cost

**Criteria:**
1. Each worker must start from and end at the HHC center, i.e., all of the routes must start and end at the same node.
2. Each elder may require more than one service per day.
3. Each service has an appointed time period which forbids any caregiver to begin service after the lower bound, and to wait when they arrive before the upper bound.
4. Each caregiver is expected to finish the route within a given maximal workload.
5. Each caregiver moves among elders by a constant speed, thereby the traveling cost becomes a value only related to the distance.
6. Three rules of assigning demands to caregivers must be obeyed:
    6.1. Each caregiver masters a set of skill qualifications, and each service has a set of needed qualifications. An elder has a set of needed services. It is assumed that, only when a job’s qualification set is a sub-set of a caregiver’s qualification set, can the caregiver serve the elder (hierarchical matching).
    6.2. Each pair of caregiver and elder has an initial preference level, indicating the degree to which they mutually prefer each other.
    6.3. Once a caregiver is assigned to an elder who requires more than one service, the caregiver is expected to execute the rest of the services of the elder, unless the caregiver is not qualified enough or unable to obey other constraints.
7. Service time is defined as normally distributed such that the mean value is linked to the three rules defined in criterion 6. 
    7.1 Firstly, the skill-demand-match constraint is negatively correlated to the mean value, i.e., the better the skills-match quality is, the lesser service time is predicted.
    7.2 Secondly, the preference level is defined as a weight of the mean value, i.e., the more they prefer each other, the smaller the service time is predicted.
    7.3 Thirdly, the acquaintanceship is defined as an increment of the weight defined by 9.2, i.e., as a caregiver and an elder become more and more familliar with each other, the service time is predicted to be lesser and lesser.


## Mathematical model.

The problem is hierarchically modeled by the MDPs as the exterior level and the CCP-MOOP model as the interior level such that the CCP-MOOP model handles every criterion mentioned above but only generates one route each time, and the MDPs represent a process during which a nurse will be chosen as an input to the CCP-MOOP model at each state. Therefore, every single episode of the MDPs contains three events: 1) observe the current state which is practically some information about the available targets to visit; 2) take an action which is practically a nurse; 3) observe the reward according to the solution given by the algorithm that solves the CCP-MOOP model.

