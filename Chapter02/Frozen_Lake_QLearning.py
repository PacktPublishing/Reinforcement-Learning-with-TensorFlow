
# coding: utf-8

# In[1]:

from __future__ import print_function
import gym
import numpy as np
import time



# ## Load the environment

# In[2]:


env = gym.make('FrozenLake-v0')


# In[5]:


s = env.reset()
print(s)


# In[6]:


env.render()


# In[7]:


print(env.action_space)   #number of actions


# In[8]:


print(env.observation_space)  #number of states


# In[9]:


print("Number of actions : ",env.action_space.n)
print("Number of states : ",env.observation_space.n)


# ## Epsilon Greedy

# In[40]:


def epsilon_greedy(Q,s,na):
    epsilon = 0.3
    p = np.random.uniform(low=0,high=1)
    #print(p)
    if p > epsilon:
        return np.argmax(Q[s,:])#say here,initial policy = for each state consider the action having highest Q-value
    else:
        return env.action_space.sample()


# ## Q-Learning Implementation

# In[47]:


#Initializing Q-table with zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])

#set hyperparameters
lr = 0.5 #learning rate
y = 0.9 #discount factor lambda
eps = 100000 #total episodes being 100000


for i in range(eps):
    s = env.reset()
    t = False
    while(True):
        a = epsilon_greedy(Q,s,env.action_space.n)
        s_,r,t,_ = env.step(a)
        if (r==0):  
            if t==True:
                r = -5 #to give negative rewards when holes turn up
                Q[s_] = np.ones(env.action_space.n)*r    #in terminal state Q value equals the reward
            else:
                r = -1  #to give negative rewards to avoid long routes
        if (r==1):
                r = 100
                Q[s_] = np.ones(env.action_space.n)*r    #in terminal state Q value equals the reward
        Q[s,a] = Q[s,a] + lr * (r + y*np.max(Q[s_,a]) - Q[s,a])
        s = s_   
        if (t == True) :
                break


# In[48]:


print("Q-table")
print(Q)


# In[49]:


s = env.reset()
env.render()
while(True):
    a = np.argmax(Q[s])
    s_,r,t,_ = env.step(a)
    env.render()
    s = s_
    if(t==True) :
        break


# In[ ]:




