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
   - Each caregiver masters a set of skill qualifications, and each service has a set of needed qualifications. An elder has a set of needed services. It is assumed that, only when a job’s qualification set is a sub-set of a caregiver’s qualification set, can the caregiver serve the elder (hierarchical matching).
   - Each pair of caregiver and elder has an initial preference level, indicating the degree to which they mutually prefer each other.
   - Once a caregiver is assigned to an elder who requires more than one service, the caregiver is expected to execute the rest of the services of the elder, unless the caregiver is not qualified enough or unable to obey other constraints.
    
7. Service time is defined as normally distributed such that the mean value is linked to the three rules defined above.
   - Firstly, the skill-demand-match constraint is negatively correlated to the mean value, i.e., the better the skills-match quality is, the lesser service time is predicted.
   - Secondly, the preference level is defined as a weight of the mean value, i.e., the more they prefer each other, the smaller the service time is predicted.
   - Thirdly, the acquaintanceship is defined as an increment of the weight defined by 9.2, i.e., as a caregiver and an elder become more and more familliar with each other, the service time is predicted to be lesser and lesser.

## Mathematical model.

The problem is hierarchically modeled by the MDPs as the exterior level, and the CCP-MOOP model as the interior level such that the CCP-MOOP model represents the criteria given above but only cares for one single route each time, and the MDPs represents a process during which a caregiver will be chosen as an input to the CCP-MOOP model at each state. Therefore, before it reaches the absorbing state which is either out of caregiver or out of demand, every single episode of the MDPs contains three events: 
   - observe the current state which is practically some information about the available targets to visit.
   - take an action, that is, choose a caregiver.
   - observe the reward according to the solution given by the algorithm that solves the CCP-MOOP model.

**Finite Markov Decision Processes**
1. The state set is defined according to the size-relationship between every two levels of the demands. Only two relationships are considered: ">=" and "<". For example, 6 sorts of relationships exist among 3 demand levels, and with 2 absorbing states representing the situations of being out of caregiver or demand there will be 8 states in total.
   - *0 d1 >= d2 >= d3*
   - *1 d1 >= d3 >= d2*
   - *2 d2 >= d1 >= d3*
   - *3 d2 >= d3 >= d1*
   - *4 d3 >= d1 >= d2*
   - *5 d3 >= d2 >= d1*
   - *6 no more demands*
   - *7 no more nurses*

2. The action set depends on the instance that provides caregivers with different numbers at different skill levels.
   - *1, 2, 3, ..., n*
   
3. The transition probability is not necessary to be precisely given since we will be applying Q Learning which is model free.

4. The reward function is defined by the scalarisation of the two objectives. 
   - *r = fd - ln(twt^6) + 50)*
   - *in which **fd** is the number of fulfilled demands of the sub-solution, and **twt** is the total waiting time of the sub-solution.*
   - ***note that, the sub-solution is a route given by the algorithm that solves the CCP-MOOP.***
   
**Chance Constrained Programming & Multi-objective Optimisation Programming**
1. Objective functions are scalarised, thereby the MOOP is transformed into a single objective optimization problem.
   - *Max 
