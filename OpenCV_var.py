
# coding: utf-8

# In[1]:


import cv2


# In[2]:


import numpy as np
import pandas as pd
import os
import pickle


#from keras.models import Sequential
#from keras.layers.convolutional import Convolution2D
#from keras.layers.pooling import MaxPooling2D
#from keras.layers.core import Flatten, Dense, Activation, Dropout
from keras.preprocessing import image
#from keras.callbacks import EarlyStopping, ModelCheckpoint
#from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score


input_size = (512, 512)

# In[ ]:
'''

X = []
y = []
y_pred = []

threshold = 600.0


# In[ ]:


t = []
t1 = []


# In[ ]:


folderpath = 'CERTH_ImageBlurDataset/TrainingSet/Undistorted/'

# load image arrays
for filename in os.listdir(folderpath):
    if filename != '.DS_Store':
        imagepath = folderpath + filename
        img = image.load_img(imagepath, target_size=(192,192))
        #X.append(np.asarray(img))
        #image = cv2.imread(imagepath)
        gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2GRAY)
        fm = variance_of_laplacian(gray)
        t1.append(fm)

        if fm < threshold:
            y_pred.append(1)
        else:
            y_pred.append(0)
        y.append(0)
    else:
        print(filename, 'not a pic')


# In[ ]:


folderpath = 'CERTH_ImageBlurDataset/TrainingSet/Artificially-Blurred/'

# load image arrays
for filename in os.listdir(folderpath):
    if filename != '.DS_Store':
        imagepath = folderpath + filename
        img = image.load_img(imagepath, target_size=(192,192))
        #X.append(np.asarray(img))
        gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2GRAY)
        fm = variance_of_laplacian(gray)
        t.append(fm)
        if fm < threshold:
            y_pred.append(1)
        else:
            y_pred.append(0)
        t.append(fm)
        y.append(1)
    else:
        print(filename, 'not a pic')


# In[ ]:


folderpath = 'CERTH_ImageBlurDataset/TrainingSet/Naturally-Blurred/'

# load image arrays
for filename in os.listdir(folderpath):
    if filename != '.DS_Store':
        imagepath = folderpath + filename
        img = image.load_img(imagepath, target_size=(192,192))
        #X.append(np.asarray(img))
        gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2GRAY)
        fm = variance_of_laplacian(gray)
        t.append(fm)
        if fm < threshold:
            y_pred.append(1)
        else:
            y_pred.append(0)
        t.append(fm)
        y.append(1)33
    else:
        print(filename, 'not a pic')


# In[3]:
'''

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()



# In[ ]:



# In[ ]:


#accuracy_score(y, y_pred)


# In[4]:


X_test = []
y_test = []
y_pred = []
t_blur = []
t_nblur = []
threshold = 400


# In[5]:


dgbset = pd.read_excel('CERTH_ImageBlurDataset/EvaluationSet/DigitalBlurSet.xlsx')
nbset = pd.read_excel('CERTH_ImageBlurDataset/EvaluationSet/NaturalBlurSet.xlsx')


# In[6]:


dgbset['MyDigital Blur'] = dgbset['MyDigital Blur'].apply(lambda x : x.strip())
dgbset = dgbset.rename(index=str, columns={"Unnamed: 1": "Blur Label"})
nbset['Image Name'] = nbset['Image Name'].apply(lambda x : x.strip())


# In[7]:


folderpath = 'CERTH_ImageBlurDataset/EvaluationSet/DigitalBlurSet/'

# load image arrays
for filename in os.listdir(folderpath):
    if filename != '.DS_Store':
        imagepath = folderpath + filename
        img = image.load_img(imagepath, target_size= input_size)
        #X_test.append(np.asarray(img))
        blur = dgbset[dgbset['MyDigital Blur'] == filename].iloc[0]['Blur Label']
        gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2GRAY)
        fm = variance_of_laplacian(gray)
        if fm < threshold:
            y_pred.append(1)
        else:
            y_pred.append(0)

        if blur == 1:
            y_test.append(1)
            t_blur.append(fm)
        else:
            y_test.append(0)
            t_nblur.append(fm)
    else:
        print(filename, 'not a pic')

print("Artificially Blurred Evaluated...")
# In[8]:


folderpath = 'CERTH_ImageBlurDataset/EvaluationSet/NaturalBlurSet/'

# load image arrays
for filename in os.listdir(folderpath):
    if filename != '.DS_Store':
        imagepath = folderpath + filename
        img = image.load_img(imagepath, target_size=input_size)
        #X_test.append(np.asarray(img))
        blur = nbset[nbset['Image Name'] == filename.split('.')[0]].iloc[0]['Blur Label']
        gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2GRAY)
        fm = variance_of_laplacian(gray)
        if fm < threshold:
            y_pred.append(1)
        else:
            y_pred.append(0)
        if blur == 1:
            y_test.append(1)
            t_blur.append(fm)
        else:
            y_test.append(0)
            t_nblur.append(fm)
    else:
        print(filename, 'not a pic')

print("Naturally Blurred Evaluated...")


# In[9]:
with open('y_test.pkl', 'rb') as picklefile:
    y_test = pickle.load(picklefile)


accuracy = accuracy_score(y_test, y_pred)
print("accuracy: {:.2f}%".format(accuracy * 100))

## Accuracy: 87.29%
