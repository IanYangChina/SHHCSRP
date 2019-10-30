# An algorithm hibridized by Q-learning and Ant Colony Optimisation to solve the SHHCSRP

The project is my first attempt to build a tabular Q-learning agent to solve a complicated and large scale combinatorial optimisation problem. The problem, **Stochastic Home Health Care Scheduling and Routing Problem**, is a variant of the well-known Stochastic Vehicle Routing Problem (SVRP). Basically, the task of such kind of problems is to assign a group of demands required by many customers to a group of vehicles with optimised cost. This project aims to train an agent that makes its own decisions and generate a routing plan. 

**This repo is academic-oriented.**

## Brief Introduction

The SVRP has been studied for many decades and now there is a substantial amount of papers with various applications, theories, mathematical models, objectives and constraints. As for this project, I intended to use the Q-learning technique and see if this popular RL algorithm can obtain satisfactory performance on large scale stochastic combinatorial problem.

Since RL algorithm requires a MDPs, the SVRP in this project was hierarchically modeled by the MDPs as the exterior level, and the CCP model as the interior level such that the CCP model represents the criteria of a single vehicle routing process, and the MDPs represents a process during which a vehicle will be chosen as an input to the CCP model at each state. Before it reaches the absorbing state which is either out of vehicle or out of demand, every single episode of the MDPs contains three steps: 
   - observe the current state which is practically some information about the available targets to visit.
   - take an action, that is, choose a vehicle.
   - observe the reward according to the solution given by the algorithm that solves the CCP model.

The objective of the model is to maximize the fulfilled demands with least waiting cost. The QL agent was expected to learn to make a plan that satisfies the objective as much as possible. 

But till now the algorithm does yet perform very well on the instance, which is probably due to the following problems:

1. The state definition of the MDPs is not correct enough that it couldn't fully represent the environment.
2. The reward function of the MDPs is not appropriate enough that it couldn't lead the agent towords sensible policy.
3. The ACO algorithm that solves the CCP model is essentially based on random evolution, which commonly produces different sub-solution at each state of the MDPs, thus makes it more difficult for the QL agent to learn the right behavious.
4. Unsupervised algorithms (like CNN, DQN, QL, etc.) need a huge amount of training data to learn sensible results. 

However, this project was one part of my last research subject that is now finished, and my first motivation was the curiosity of how well the Reinforcement Learning techniques could preform on large scale combinatorial optimisation problem. The reason I put it on Github is to make it as a reference that might be useful for someone who wants to make further investigation on this direction or take it as a practice of implementing Q-learning on combinatorial optimisation problem. Here are some suggestions that might help this project to make further progress:

1. Better shape the reward function. 
2. Since the state difiniton here is highly abstracted from the information of resource and demand, which turned out to be insufficient for the QL agent to learn more important knowledge from the environment, one may consider figuring out a better approximation or abstraction of the environment.

In spite of what's mentioned above, there are plenty of directions that this project could move towards. Besides, I hope you can learn something useful from this repo :)
For detailed information please check out the Wiki pages: [Wiki](https://github.com/IanYangChina/Q-learning---Ant-colony-optimization/wiki)

## Files explanation
1. The 2 MS Visio files (.vsdx) are flow charts of the BWACO and QL algorithms.
2. The 4 MS Excel files (.xlsx) are instances data.
3. The 2 CSV files (.csv) are initial preference settings for relevant instances.
4. [QL_BWACO.py](https://github.com/IanYangChina/SHHCSRP/blob/master/QL_BWACO.py) is the Python code for algorithms
5. [main.py](https://github.com/IanYangChina/SHHCSRP/blob/master/main.py) is the main py file to run.

## Run
1. Setup expermental instance and nurse resource which are at the beginning in the "__main__".
2. Run 'main.py'.
3. After training, find new files in the root folder: things like **solution** with extensions like **.png** and **.xls**. They are results.
