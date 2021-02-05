#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
import random


# In[3]:


#Create enviornment using gym

env = gym.make("Taxi-v3")


# In[4]:


env.render()


# In[5]:


alpha = 0.4
gamma = 0.999
epsilon = 0.017

q = {}
for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        q[(s,a)]=0


# In[6]:


def update_q_table(prev_state,action,reward,nextstate,alpha,gamma):
    qa = max([q[(nextstate,a)]for a in range(env.action_space.n)])
    q[(prev_state,action)] += alpha*(reward+gamma*qa-q[(prev_state,action)])


# In[7]:


def epsilon_greedy_policy(state,epsilon):
    if random.uniform(0,1) < epsilon:
        return env.action_space.sample()
    else:
        return max(list(range(env.action_space.n)),key= lambda x: q[(state,x)])


# In[ ]:


for i in range(8000):
    r = 0
    prev_state = env.reset()
    
    while True:
        env.render()
        
        #In each state we select action by epsilon greedy algo
        action = epsilon_greedy_policy(prev_state,epsilon)
        
        #then we perform action and move to next state
        #received the reward
        nextstate, reward, done, _ = env.step(action)
        
        #Next we update Q value
        
        update_q_table(prev_state,action,reward,nextstate,alpha,gamma)
        
        #Finally we update previous state as next state
        prev_state = nextstate
        
        #Store all the rewards obtained
        r += reward
        
        #we will break the loop if we are at the terminal
        #state the episode
        if done:
            break
            
    print("Total Rewards:",r)
    
env.close()
            


# In[ ]:




