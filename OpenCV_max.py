
# coding: utf-8

# In[6]:


import cv2


# In[7]:


import numpy as np
import pandas as pd
import os
import pickle

from sklearn.metrics import accuracy_score


# In[8]:


def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()





X_test = []
y_test = []
y_pred = []
val_blur = []
val_nblur = []
threshold = 215
input_size = (192, 192)


# In[10]:


dgbset = pd.read_excel('CERTH_ImageBlurDataset/EvaluationSet/DigitalBlurSet.xlsx')
nbset = pd.read_excel('CERTH_ImageBlurDataset/EvaluationSet/NaturalBlurSet.xlsx')


# In[11]:


dgbset['MyDigital Blur'] = dgbset['MyDigital Blur'].apply(lambda x : x.strip())
dgbset = dgbset.rename(index=str, columns={"Unnamed: 1": "Blur Label"})
nbset['Image Name'] = nbset['Image Name'].apply(lambda x : x.strip())


# In[19]:


folderpath = 'CERTH_ImageBlurDataset/EvaluationSet/DigitalBlurSet/'

# load image arrays
for filename in os.listdir(folderpath):
    if filename != '.DS_Store':
        imagepath = folderpath + filename
        blur = dgbset[dgbset['MyDigital Blur'] == filename].iloc[0]['Blur Label']
        gray = cv2.resize(cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE), input_size)
        X_test.append(gray)
        maxval = np.max(cv2.convertScaleAbs(cv2.Laplacian(gray,3)))
        if maxval < threshold:
            y_pred.append(1)
        else:
            y_pred.append(0)

        if blur == 1:
            y_test.append(1)
            val_blur.append(maxval)
        else:
            y_test.append(0)
            val_nblur.append(maxval)
    else:
        print(filename, 'not a pic')

print("Digitally Blurred Evaluated...")

# In[ ]:


folderpath = 'CERTH_ImageBlurDataset/EvaluationSet/NaturalBlurSet/'

# load image arrays
for filename in os.listdir(folderpath):
    if filename != '.DS_Store':
        imagepath = folderpath + filename
        blur = nbset[nbset['Image Name'] == filename.split('.')[0]].iloc[0]['Blur Label']
        gray = cv2.resize(cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE), input_size)
        X_test.append(gray)
        maxval = np.max(cv2.convertScaleAbs(cv2.Laplacian(gray,3)))
        if maxval < threshold:
            y_pred.append(1)
        else:
            y_pred.append(0)
        if blur == 1:
            y_test.append(1)
            val_blur.append(maxval)
        else:
            y_test.append(0)
            val_nblur.append(maxval)
    else:
        print(filename, 'not a pic')

print("Naturally Blurred Evaluated...")

# In[ ]:


accuracy = accuracy_score(y_test, y_pred)
print("accuracy: {:.2f}%".format(accuracy * 100))


# In[ ]:


from scipy import stats
print(np.percentile(val_blur,60))
print(np.percentile(val_nblur,15))

print(stats.describe(val_blur))
print(stats.describe(val_nblur))


# In[ ]:
