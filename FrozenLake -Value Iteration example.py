#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym


# In[2]:


import numpy as np


# In[3]:


env = gym.make('FrozenLake-v0')


# In[4]:


env.observation_space.n


# In[5]:


env.action_space.n


# In[6]:


def value_iteration(env, gamma = 1.0):
    value_table = np.zeros(env.observation_space.n)
    no_of_iterations = 10000
    threshold = 1e-20
    
    for i in range(no_of_iterations):
        update_value_table = np.copy(value_table)
        for state in range(env.observation_space.n):
            Q_value = []
            for action in range(env.action_space.n):
                next_states_reward = []
                for next_sr in env.P[state][action]:
                    trans_prob, next_state, reward_prob, _ = next_sr
                    next_states_reward.append((trans_prob*(reward_prob+gamma*update_value_table[next_state])))
                Q_value.append(np.sum(next_states_reward))
            value_table[state] = max(Q_value)
        if (np.sum(np.fabs(update_value_table - value_table)) <= threshold):
            print('Value-iteration converged at iteration# %d' %(i+1))
            break
    return value_table
    


# In[7]:


opt_value_function = value_iteration(env=env,gamma=1.0)


# In[10]:


def extract_policy(value_table, gamma = 1.0):
    policy = np.zeros(env.observation_space.n)
    for state in range(env.observation_space.n):
        Q_table = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for next_sr in env.P[state][action]:
                trans_prob, next_state, reward_prob, _ = next_sr
                Q_table[action] += (trans_prob*(reward_prob+gamma*value_table[next_state]))
        policy[state] = np.argmax(Q_table)
    return policy 


# In[11]:


optimal_policy = extract_policy(opt_value_function, gamma=1.0)


# In[12]:


print(optimal_policy)


# In[ ]:



        

