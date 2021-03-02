#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Numpy library

import numpy as np


# In[2]:


first_numpy_array = np.array([1,2,3,4])


# In[3]:


print(first_numpy_array)


# In[4]:


#array of zeros

array_with_zeros = np.zeros((3,3))
array_with_zeros


# In[5]:


#array with ones
array_with_ones = np.ones((3,3))
array_with_ones


# In[6]:


#array_with empty

array_with_empty = np.empty((2,3))


# In[7]:


array_with_empty


# In[8]:


#arange function

np_arange = np.arange(12)


# In[9]:


np_arange


# In[10]:


np_arange.reshape(3,4)


# In[11]:


#linspac function for equally spaced data elements

np_linpce = np.linspace(1,6,4)


# In[12]:


np_linpce


# In[13]:


oneD_array = np.arange(15)


# In[14]:


oneD_array


# In[15]:


twoD_array = oneD_array.reshape(3,5)


# In[16]:


twoD_array


# In[17]:


oneD_array = np.arange(27)


# In[18]:


threeD_array = oneD_array.reshape(3,3,3)


# In[19]:


threeD_array


# In[20]:


#Vector addition

#We have 4 cyclist riding a distance

c1 = [10,15,17,26]
c2 = [12,11,21,24]

np_c1 = np.array(c1)
np_c2 = np.array(c2)


# In[21]:


np_c1+np_c2


# In[22]:


#Basic mathematical ops

#add

np.add(45,67)


# In[23]:


#subtract

np.subtract(45,12)


# In[24]:


#ndarray multiplcation

hours_worked = np.array([12,10,5,8])
hourly_rate = 15

total_earning = hours_worked*hourly_rate


# In[25]:


total_earning


# In[26]:


sum(total_earning)


# In[27]:


#Comparision operation

np_weekly_hrs = np.array([23,41,55,47,38])

np_weekly_hrs[np_weekly_hrs>40]


# In[28]:


np_weekly_hrs[np_weekly_hrs!=40]


# In[29]:


#Logical AND Operation

np.logical_and(np_weekly_hrs>20,np_weekly_hrs<50)


# In[30]:


#Logical Not operation

np.logical_not(np_weekly_hrs>35)


# In[38]:


#Accessing Array Elemnts and index

ct = np.array([[13,15,17,19],[14,22,21,26]])
ct


# In[32]:


first_trail_data = ct[0]


# In[33]:


first_trail_data


# In[34]:


second_trail_data = ct[1]


# In[35]:


second_trail_data


# In[36]:


#Accessing the first element in the array

ct[0][0]


# In[37]:


#to access elemts from both row and 1st column

ct[:,0]


# In[39]:


ct.shape


# In[40]:


#Slicing
ct[:,1:3]


# In[41]:


#iterate through dataset

for recs in ct:
    print(recs)


# In[42]:


for recs in ct[:,1:3]:
    print(recs)


# In[43]:


#Indexing with Boolean Array

test_scores = np.array([[83,71,45,63],[54,56,88,82]])


# In[44]:


passing_scores = test_scores>60


# In[45]:


passing_scores


# In[46]:


test_scores[passing_scores]


# # Copy and View

# In[47]:


Cities = np.array(['Manhattan','Adelaid','Berlin','New Jersy','Delhi'])


# In[48]:


Cities


# In[49]:


C_Cities = Cities


# In[50]:


C_Cities


# In[51]:


C_Cities is Cities


# In[52]:


View_of_Cities = Cities.view()


# In[53]:


View_of_Cities


# In[54]:


View_of_Cities[4] = 'Chennai'


# In[55]:


View_of_Cities


# In[56]:


Cities


# In[57]:


Copy_of_Cities = Cities.copy()


# In[58]:


Copy_of_Cities


# In[59]:


Copy_of_Cities[4] = 'Mumbai'


# In[60]:


Copy_of_Cities


# In[61]:


Cities


# In[63]:


#Broadcasting 

import numpy as np

#Create two array of same shape
array_a = np.array([2,3,4,5])
array_b = np.array([.2,.3,.34,.5])


# In[64]:


array_a*array_b


# In[65]:


scaler_c = .3


# # Transpose

# In[68]:


test_scores = np.array([[83,71,45,63],[54,56,88,82]])
test_scores


# In[69]:


test_scores.transpose()


# In[70]:


inverse_array = np.array([[10,20],[15,25]])
np.linalg.inv(inverse_array)


# In[71]:


trace_array = np.array([[10,20],[22,32]])
np.trace(trace_array)


# In[ ]:




