# An algorithm hibridized by Q-learning and Ant Colony Optimisation to solve the SVRP

The project is my first attempt to build a reinforcement learning agent for solving complicated and large scale combinatorial optimisation problem. The target problem chosen for this project is a variant of the well-known Stochastic Vehicle Routing Problem with some complicated constraints. Basically, the task of such kind of problem is to assign a group of demands required by many customers to a group of vehicles with optimised cost, and this project aims to train an agent to make its own decisions generating a vehicle schedule plan.

A statement here to make is that this repo is more like a tutorial of Q-learning, Ant Colony Optimisation, SVRP and Python but an appealing or applicable realisation for real-world application or academic research. It might disappoint those who are expecting something from this aspects. But for those who are seeking references or learning/practicing examples, this might be a good one. :)

P.S. This is also a summary of my last study stage.

## What you might possibly learn or get from this repo:

1. What is Stochastic Vehicle Routing Problem
2. What is Markov Decision Processes and how it works
3. What is Chance Constained Programming and how it works
4. What is Q-learning and how to realize it by Python
5. What is Ant Colony Optimisation and how to realize it by Python

## Brief Introduction

The SVRP has been studied for many decades and now there is a substantial amount of papers with various applications, theories, mathematical models, objectives and constraints. As for this project, I intended to use the Q-learning technique and see if this kind of Reinforcement Learning algorithm can obtain satisfactory performance on large scale stochastic combinatorial problem.

Therefore, the SVRP in this project is hierarchically modeled by the MDPs as the exterior level, and the CCP model as the interior level such that the CCP model represents the criteria of a single vehicle routing process, and the MDPs represents a process during which a vehicle will be chosen as an input to the CCP model at each state. Before it reaches the absorbing state which is either out of vehicle or out of demand, every single episode of the MDPs contains three steps: 
   - observe the current state which is practically some information about the available targets to visit.
   - take an action, that is, choose a vehicle.
   - observe the reward according to the solution given by the algorithm that solves the CCP model.

Two objectives of the model are defined as: 1) to maximize the fulfilled demands, and 2) to minimize the total waiting time. The QL agent was expected to learn how to make a plan that satisfies the objectives as much as possible. 

But till now the algorithm does yet perform very well on the instance, which is probably due to the following problems:

1. The state definition of the MDPs is not correct enough that it couldn't fully represent the environment.
2. The reward function of the MDPs is not appropriate enough that it couldn't lead the agent towords sensible policy.
3. The ACO algorithm that solves the CCP model is essentially based on random evolution, which commonly produces different sub-solution at each state of the MDPs, thus makes it more difficult for the QL agent to learn the right behavious.

However, this project was one part of my last research subject that is now finished, and my first motivation was the curiosity of how well the Reinforcement Learning techniques could preform on discrete decision problem. The reason I put it on Github is to make it as a reference that might be useful for someone who wants to make further investigation on this direction or take it as a practice of implementing Q-learning on combinatorial optimisation problem. Here are some significant knowledge or techniques that could help this project to make further progress:

1. Two well-known approaches for solving the Multi-objective combinatorial optimisation problem: **weighted sum scalarisation and compromise solution method**. By applying these methods, one might be able to better shape the reward function of the QL agent. 
2. One can rank the objectives and solve the problem by satisfying them one after one. By doing so one can lower the complexity of the original problem albeit it actully treats the objectives unequally with **prioritisation**. But it could still be referred by real-world engineering or other fields where decision makers treat the objectives differentially.
3. Since the state difiniton here is highly abstracted from the information of resource and demand, which turned out insufficient for the QL agent to learn more important knowledge from the environment, one should consider figuring out a better approximation or abstraction of the environment. (Deep neural network might not work very well since it performs better on continuous repersentation, however it is encouraged to have a try).

In spite of things mentioned above, there are plenty of directions that this project could move towards. Besides, I hope you can learn something useful from this repo :)
For detailed information please check out the Wiki pages: [Wiki](https://github.com/IanYangChina/Q-learning---Ant-colony-optimization/wiki)

## Direct try-out
1. Clone the repo
2. Run 'main.py' by any method
3. Find new files in the root folder: things like solution*** with extensions like .png and .xls. They are results.
