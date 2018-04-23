# importing dependency libraries
from __future__ import print_function
import gym
import numpy as np
import time

#Load the environment
env = gym.make('FrozenLake-v0')

s = env.reset()
print(s)
print()

env.render()
print()

print(env.action_space) #number of actions
print(env.observation_space) #number of states
print()

print("Number of actions : ",env.action_space.n)
print("Number of states : ",env.observation_space.n)
print()

# Value Iteration Implementation

#Initializing Utilities of all states with zeros
U = np.zeros([env.observation_space.n])

#since terminal states have utility values equal to their reward
U[15] = 1 #goal state
U[[5,7,11,12]] = -1 #hole states
termS = [5,7,11,12,15] #terminal states
#set hyperparameters
y = 0.8 #discount factor lambda

eps = 1e-3 #threshold if the learning difference i.e. prev_u - U goes below this value break the learning

i=0
while(True):
	i+=1
	prev_u = np.copy(U)
	for s in range(env.observation_space.n):
		q_sa = [sum([p*(r + y*prev_u[s_]) for p, s_, r, _ in env.env.P[s][a]]) for a in range(env.action_space.n)]
		if s not in termS: 
			U[s] = max(q_sa)
	if (np.sum(np.fabs(prev_u - U)) <= eps):
		print ('Value-iteration converged at iteration# %d.' %(i+1))
		break

print("After learning completion printing the utilities for each states below from state ids 0-15")
print()
print(U[:4])
print(U[4:8])
print(U[8:12])
print(U[12:16])
