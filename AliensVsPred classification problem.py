#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(' wget https://bashupload.com/RwEtV/tDdQ-.zip')


# In[2]:


get_ipython().system(' unzip ALandPData.zip')


# In[3]:


# Data Augmentation

from tensorflow.keras.preprocessing.image import ImageDataGenerator

augmenter = ImageDataGenerator(rotation_range=30,  #Rotation
                              zoom_range=0.2,      #Zooming
                              horizontal_flip=True, #Horizontal Flip
                              fill_mode='nearest')


# In[8]:


imageGen = augmenter.flow_from_directory('AlienDataset/',
                         batch_size=1,
                         save_to_dir="AlienAugDataset/", #Ensure the directory exists else prog will throw error
                         save_prefix="lion-",
                         save_format="jpg")


counter = 0

for generatedImage in imageGen:
    
    counter += 1 
    
    if counter == 100:
        break


# In[9]:


imageGen = augmenter.flow_from_directory('PredDataset//',
                         batch_size=1,
                         save_to_dir="PredAugDataset/", #Ensure the directory exists else prog will throw error
                         save_prefix="lion-",
                         save_format="jpg")


counter = 0

for generatedImage in imageGen:
    
    counter += 1 
    
    if counter == 100:
        break


# In[11]:


import tensorflow as tf


# In[12]:


train_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.)
test_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.)


# In[13]:


#Pass the images to generator an input compatible for CNN

trainImageData = train_generator.flow_from_directory('DatasetALandP/data/train',batch_size=20,class_mode='binary',target_size=(64,64))


# In[14]:


#Pass the images to generator an input compatible for CNN - Testdata
 
testImageData = test_generator.flow_from_directory('DatasetALandP/data/validation',batch_size=20,class_mode='binary',target_size=(64,64))


# In[15]:


#Check shapes of training and testing data

print(trainImageData.image_shape)
print(testImageData.image_shape)


# Architect a CNN layer

# In[16]:


model = tf.keras.models.Sequential()

#Create 1st Convolutional layer
model.add(tf.keras.layers.Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu',padding='same'))

#pooling Operation

model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2)))

#Create 2nd Convolutional layer
model.add(tf.keras.layers.Conv2D(16,(3,3),input_shape=(64,64,3),activation='relu',padding='same'))

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))


# In[17]:


# Array Flattening

model.add(tf.keras.layers.Flatten())

#Add Fully connected layers

model.add(tf.keras.layers.Dense(units=512,activation='relu'))
model.add(tf.keras.layers.Dense(units=512,activation='relu'))
model.add(tf.keras.layers.Dense(units=512,activation='relu'))
model.add(tf.keras.layers.Dense(units=512,activation='relu'))

#Add Output layer

model.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))


# In[18]:


#Compliation

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[19]:


#How many images we have in  single batch
int(len(trainImageData.filenames)/trainImageData.batch_size)


# In[20]:


model.fit(trainImageData,validation_data=testImageData,epochs=50,steps_per_epoch=int(len(trainImageData.filenames)/trainImageData.batch_size)
,validation_steps = int(len(testImageData.filenames)/testImageData.batch_size))


# In[21]:


img = tf.keras.preprocessing.image.load_img('DatasetALandP/data/train/alien/103.jpg',target_size=(64,64))


# In[23]:


trainImageData.class_indices


# In[24]:


#Create an image array

imgArray = tf.keras.preprocessing.image.img_to_array(img)
imgArray.shape


# In[25]:


#Make Image compatible for input

import numpy as np
compatibleImgArray = np.expand_dims(imgArray,axis=0)

compatibleImgArray.shape


# In[26]:


#Predict the class

if model.predict_classes(compatibleImgArray)==0:
    print("Its an alien,you need to run now....")
else:
    print("Its a predator,Hide I mean what else you can do... ")
    
    
    


# In[ ]:





# In[ ]:




