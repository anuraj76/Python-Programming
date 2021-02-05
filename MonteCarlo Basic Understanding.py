#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Estimate value of Pi

import numpy as np
import math
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Now lets initialze square size and no. of points inside square and cirlce

square_size = 1
points_inside_circle = 0
points_inside_sqaure = 0
sample_size = 1000
arc = np.linspace(0,np.pi/2,100)


# In[3]:


#define the fuction that generates random points inside square

def generate_points(size):
    x = random.random()*size
    y = random.random()*size
    return(x,y)


# In[4]:


#define a function to check if a point falls within circle
def is_in_circle(point,size):
    return math.sqrt(point[0]**2+point[1]**2)<=size


# In[6]:


#define a function for calculating pi value

def compute_pi(points_inside_circle,points_inside_sqaure):
    return 4*(points_inside_circle/points_inside_sqaure)


# In[8]:


plt.axes().set_aspect('equal')
plt.plot(1*np.cos(arc), 1*np.sin(arc))

for i in range(sample_size):
    point = generate_points(square_size)
    plt.plot(point[0],point[1],'c.')
    points_inside_sqaure +=1
    if is_in_circle(point,square_size):
        points_inside_circle +=1


# In[9]:


print('Apprx value of pi is {}'.format(compute_pi(points_inside_circle,points_inside_sqaure)))


# In[ ]:




