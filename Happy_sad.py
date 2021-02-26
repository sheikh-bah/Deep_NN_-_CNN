#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import zipfile


# We extract our files 

# In[46]:


zip_file = 'C:/Users/sheikh/Downloads/happy-or-sad.zip'
zip_ref = zipfile.ZipFile(zip_file, 'r')
zip_ref.extractall('C:/Users/sheikh/Downloads/happy-or-sad')
zip_ref.close()
            
                
        
# using Callback
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = {}):
        if (logs.get('accuracy') > 0.999):
            print("\nReached 99% accuracy, stop training")
            self.model.stop_training = True
Callbacks = MyCallback()


# In[47]:


train_happy = os.path.join('C:/Users/sheikh/Downloads/happy-or-sad', 'happy')
train_sad = os.path.join('C:/Users/sheikh/Downloads/happy-or-sad', 'sad')


# In[48]:


train_happy_list = os.listdir(train_happy)


# In[49]:


len(train_happy_list)


# In[50]:


train_sad_list = os.listdir(train_sad)


# In[51]:


len(train_sad_list)


# # We build the model

# In[52]:


import tensorflow as tf
from tensorflow import keras


# In[53]:


model = keras.models.Sequential([
    # first convolution
    keras.layers.Conv2D(64, (3,3), activation = 'relu',input_shape = (150,150, 3)),
    keras.layers.MaxPooling2D(2,2),
    #second convolution
    keras.layers.Conv2D(32, (3,3), activation = 'relu'),
    keras.layers.MaxPooling2D(2,2),
    # third convolution
    keras.layers.Conv2D(16, (3,3), activation = 'relu'),
    keras.layers.MaxPooling2D(2,2),
    # FLATTEN THE RESULT INTO THE DNN
    keras.layers.Flatten(),
    # 512 hidden neuron
    keras.layers.Dense(512, activation = 'relu'),
    # only 1 output neuron
    keras.layers.Dense(1, activation = 'sigmoid')
    
    
])


# In[54]:


model.summary()


# In[55]:


from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer = RMSprop(lr = 0.001), loss = 'binary_crossentropy', metrics = ['accuracy'])


# # image generator
# when the features are in different location.
# The image generator will create a feeder for our images and auto label them for us

# In[56]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[57]:


# we rescale our image

train_data_gen = ImageDataGenerator(rescale = 1/255)
# we will call flow from the directory to load images
train_generator = train_data_gen.flow_from_directory(
    'C:/Users/sheikh/Downloads/happy-or-sad',
    target_size = (150, 150),
    batch_size = 8,
    class_mode = 'binary')
    


# # Train the model

# In[58]:


history = model.fit(train_generator, steps_per_epoch = 4, epochs = 20, verbose = 1, callbacks = [Callbacks])


# In[ ]:




