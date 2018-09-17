# A hybridized algorithm for the HHCSRP
The project aims to find a acceptable solution for a multi-nurse, multi-appointment home health care routing problem with stochastic service time and dense locations by Q Learning and Ant Colony Optimisation algorithm. This introduction is separated into three parts, including problem description, algorithm and discussion.

## Problem description.

The problem is formally named Home Health Care Scheduling and Routing Problem (HHCSRP) and has been studied for decades since 1998 (Cheng E et al 1998). It is actually a variant of the well-known Vehicle Routing Problem with some complicated constraints. Basically, the task is to assign a group of demands required by many elders to a group of caregivers with optimised cost. However, mathematical models, their objectives and constraints differ from research to research. As for this project, the model is a hybrid one that includes finite markov decision process (MDPs), chance constrained programming (CCP), and multi-objective optimisation programming (MOOP).

**Objectives:**
1. Maximize the fulfilled demands
2. Minimize the total waiting cost

## Mathematical model.

The problem is hierarchically modeled by the MDPs as the exterior level, and the CCP-MOOP model as the interior level such that the CCP-MOOP model represents the criteria given above but only cares for one single route each time, and the MDPs represents a process during which a caregiver will be chosen as an input to the CCP-MOOP model at each state. Therefore, before it reaches the absorbing state which is either out of caregiver or out of demand, every single episode of the MDPs contains three events: 
   - observe the current state which is practically some information about the available targets to visit.
   - take an action, that is, choose a caregiver.
   - observe the reward according to the solution given by the algorithm that solves the CCP-MOOP model.

