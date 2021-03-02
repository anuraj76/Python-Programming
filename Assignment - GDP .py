#!/usr/bin/env python
# coding: utf-8

# In[6]:


import zipfile


# In[17]:


import numpy as np


# In[7]:


from zipfile import ZipFile 


# In[8]:


file_name = "GDP_dataset.zip"


# In[11]:


with ZipFile('GDP_dataset.zip', 'r') as zipObj:
   # Extract all the contents of zip file in current directory
   zipObj.extractall()

    


# In[18]:


Countries = np.array(['Algeria','Angola','Argentina','Australia','Austria','Bahamas','Bangladesh','Belarus','Belgium','Bhutan','Brazil','Bulgaria','Cambodia','Cameroon','Chile','China','Colombia','Cyprus','Denmark','El Salvador','Estonia','Ethiopia','Fiji','Finland','France','Georgia','Ghana','Grenada','Guinea','Haiti','Honduras','Hungary','India','Indonesia','Ireland','Italy','Japan','Kenya', 'South Korea','Liberia','Malaysia','Mexico', 'Morocco','Nepal','New Zealand','Norway','Pakistan', 'Peru','Qatar','Russia','Singapore','South Africa','Spain','Sweden','Switzerland','Thailand', 'United Arab Emirates','United Kingdom','United States','Uruguay','Venezuela','Vietnam','Zimbabwe'
])


# In[19]:


Countries


# In[22]:


GDPs = np.array([2255.225482,629.9553062,11601.63022,25306.82494,27266.40335,19466.99052,588.3691778,2890.345675,24733.62696,1445.760002,4803.398244,2618.876037,590.4521124,665.7982328,7122.938458,2639.54156,3362.4656,15378.16704,30860.12808,2579.115607,6525.541272,229.6769525,2242.689259,27570.4852,23016.84778,1334.646773,402.6953275,6047.200797,394.1156638,385.5793827,1414.072488,5745.981529,837.7464011,1206.991065,27715.52837,18937.24998,39578.07441,478.2194906,16684.21278,279.2204061,5345.213415,6288.25324,1908.304416,274.8728621,14646.42094,40034.85063,672.1547506,3359.517402,36152.66676,3054.727742,33529.83052,3825.093781,15428.32098,33630.24604,39170.41371,2699.123242,21058.43643,28272.40661,37691.02733,9581.05659,5671.912202,757.4009286,347.7456605])


# In[23]:


GDPs


# In[24]:


max_GDP = GDPs.argmax()


# In[25]:


max_GDP


# In[26]:


Countries_with_highest_GDP = Countries[max_GDP]


# In[27]:


Countries_with_highest_GDP


# In[28]:


min_GDP = GDPs.argmin()


# In[29]:


min_GDP


# In[30]:


Country_with_lowest_GDP = Countries[min_GDP]


# In[31]:


Country_with_lowest_GDP


# In[33]:


for country in Countries:
    print('Evaluating Country {}'.format(country))


# In[34]:


for i in range(len(Countries)):
    country = Countries[i]
    country_gdp = GDPs[i]
    print('Country {} per capita gdp is {}'.format(country,country_gdp))


# In[36]:


HIghest_GDP = GDPs[max_GDP]


# In[37]:


Lowest_GDP = GDPs[min_GDP]


# In[38]:


Mean_GDP = np.mean(GDPs)


# In[39]:


Mean_GDP


# In[40]:


STD_GDP = np.std(GDPs)


# In[42]:


Summation_GDP = np.sum(GDPs)


# In[43]:


print('Highest GDP : {}, Lowest_GDP : {}, Mean_GDP : {}, STD_GDP : {}, Summation : {}'.format(HIghest_GDP,Lowest_GDP,Mean_GDP,STD_GDP,Summation_GDP))


# # Olympic Dataset

# In[47]:


oly_country = np.array(['GreatBritain'
,'China' 
,'Russia'
,'UnitedStates'
,'Korea'
,'Japan'
,'Germany'
])


# In[48]:


oly_country


# In[50]:


oly_gold = np.array([29
,38
,24
,46
,13
,7
,11
])


# In[51]:


oly_silver = np.array([17
,28
,25
,28
,8
,14
,11
])


# In[53]:


oly_bronze = np.array([19
,22
,32
,29
,7
,17
,14
])


# In[54]:


oly_bronze


# In[55]:


oly_gold


# In[56]:


oly_country[oly_gold.argmax()]


# In[57]:


Moregolds = oly_gold>20


# In[58]:


Moregolds


# In[59]:


#Find and print the countries who won more than 20 gold medals,
oly_country[Moregolds]


# In[63]:


for i in range(len(oly_country)):
    country = oly_country[i]
    gold = oly_gold[i]
    silver = oly_silver[i]
    bronze = oly_bronze[i]
    total_medals = gold+silver+bronze
    print('{}  {} {} {} {}'.format(country,gold,silver,bronze,total_medals))


# In[ ]:




