import numpy as np
import pandas as pd
import os
import pickle


from keras.preprocessing import image


# In[2]:


X_train = []
y_train = []

input_size = (96, 96)
# In[3]:


folderpath = 'CERTH_ImageBlurDataset/TrainingSet/Undistorted/'

# load image arrays
for filename in os.listdir(folderpath):
    if filename != '.DS_Store':
        imagepath = folderpath + filename
        img = image.load_img(imagepath, target_size = input_size)
        X_train.append((1/255)*np.asarray(img))
        y_train.append(0)
    else:
        print(filename, 'not a pic')
print("Trainset: Undistorted loaded...")

# In[4]:


folderpath = 'CERTH_ImageBlurDataset/TrainingSet/Artificially-Blurred/'

# load image arrays
for filename in os.listdir(folderpath):
    if filename != '.DS_Store':
        imagepath = folderpath + filename
        img = image.load_img(imagepath, target_size=input_size)
        X_train.append((1/255)*np.asarray(img))
        y_train.append(1)
    else:
        print(filename, 'not a pic')
print("Trainset: Artificially Blurred loaded...")

# In[5]:


folderpath = 'CERTH_ImageBlurDataset/TrainingSet/Naturally-Blurred/'

# load image arrays
for filename in os.listdir(folderpath):
    if filename != '.DS_Store':
        imagepath = folderpath + filename
        img = image.load_img(imagepath, target_size=input_size)
        X_train.append((1/255)*np.asarray(img))
        y_train.append(1)
    else:
        print(filename, 'not a pic')
print("Trainset: Naturally Blurred loaded...")


# Pickle the train files

with open('X_train.pkl', 'wb') as picklefile:
    pickle.dump(X_train, picklefile)


with open('y_train.pkl', 'wb') as picklefile:
    pickle.dump(y_train, picklefile)
