# Q-learning---Ant-colony-optimization
The project aims to find a acceptable solution for a multi-nurse, multi-appointment home health care routing problem with stochastic service time and dense locations by Q Learning and Ant Colony Optimisation algorithm. This introduction is separated into three parts, including problem description, algorithm and discussion.

Problem description.

The problem is formally named Home Health Care Scheduling and Routing Problem (HHCSRP) that has been studied for decades since 1998 (Cheng E et al 1998). It is actually a variant of the well-known Vehicle Routing Problem with some complicated constraints. Basically, the task is to assign a group of demands required by many elders to a group of caregivers with optimised cost. However, mathematical models, their objectives and constraints differ from research to research. As for this project, the model is a hybrid one that includes finite markov decision process, chance constrained programming, and multi-objective optimisation programming.

Objectives:
1. Maximize the fulfilled demands
2. Minimize the total waiting cost

Criteria:
1. Each worker must start from and end at the HHC center, i.e., all of the routes must start and end at the same node.
2. Each elder may require more than one service per day.
3. Each service has an appointed time period which forbids worker to begin service after the lower bound, and to wait when he/she arrives before the upper bound.
4. Each worker is expected to finish the route within a given maximal workload.
5. Each worker moves among elders by a constant speed, thereby the traveling cost becomes a value only related to the distance.
6. Each worker masters a set of skill qualifications, and each service has a set of needed qualifications. An elder has a set of needed services. It is assumed that, only when a job’s qualification set is a sub-set of a worker’s qualification set, can the worker serve the elder (hierarchical matching).
7. Once a worker is assigned to an elder who requires more than one service, the worker is expected to execute the rest of the services of the elder, unless the worker is not qualified enough or unable to obey other constraints.
8. Service time is defined as normally distributed, being linked to two of the other constraints: the 6th and 7th given above. The first one, skill-match, is negatively correlated to the mean value, i.e., the better the skills-match quality is, the lesser expected time the service needs. The second level, preference-match, is defined as a weight of the mean value, i.e., the better the match is, the smaller the expected service time is.

