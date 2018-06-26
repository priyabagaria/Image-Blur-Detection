import numpy as np
import pandas as pd
import os
import pickle


from keras.preprocessing import image


# In[2]:



input_size = (96, 96)



X_test = []
y_test = []


# In[7]:


dgbset = pd.read_excel('CERTH_ImageBlurDataset/EvaluationSet/DigitalBlurSet.xlsx')
nbset = pd.read_excel('CERTH_ImageBlurDataset/EvaluationSet/NaturalBlurSet.xlsx')


# In[8]:


dgbset['MyDigital Blur'] = dgbset['MyDigital Blur'].apply(lambda x : x.strip())
dgbset = dgbset.rename(index=str, columns={"Unnamed: 1": "Blur Label"})


# In[9]:


nbset['Image Name'] = nbset['Image Name'].apply(lambda x : x.strip())


# In[18]:


folderpath = 'CERTH_ImageBlurDataset/EvaluationSet/DigitalBlurSet/'

# load image arrays
for filename in os.listdir(folderpath):
    if filename != '.DS_Store':
        imagepath = folderpath + filename
        img = image.load_img(imagepath, target_size=input_size)
        X_test.append((1/255)*np.asarray(img))
        blur = dgbset[dgbset['MyDigital Blur'] == filename].iloc[0]['Blur Label']
        if blur == 1:
            y_test.append(1)
        else:
            y_test.append(0)
    else:
        print(filename, 'not a pic')
print("Testset: Artificially Blurred loaded...")


# In[19]:


folderpath = 'CERTH_ImageBlurDataset/EvaluationSet/NaturalBlurSet/'

# load image arrays
for filename in os.listdir(folderpath):
    if filename != '.DS_Store':
        imagepath = folderpath + filename
        img = image.load_img(imagepath, target_size=input_size)
        X_test.append((1/255)*np.asarray(img))
        blur = nbset[nbset['Image Name'] == filename.split('.')[0]].iloc[0]['Blur Label']
        if blur == 1:
            y_test.append(1)
        else:
            y_test.append(0)
    else:
        print(filename, 'not a pic')

print("Trainset: Naturally Blurred loaded...")



with open('X_test.pkl', 'wb') as picklefile:
    pickle.dump(X_test, picklefile)


with open('y_test.pkl', 'wb') as picklefile:
    pickle.dump(y_test, picklefile)
