
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import pickle


from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Flatten, Dense, Activation, Dropout
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD

# In[2]:



input_size = (96, 96)

with open('X_train.pkl', 'rb') as picklefile:
    X_train = pickle.load( picklefile)


with open('y_train.pkl', 'rb') as picklefile:
    y_train = pickle.load( picklefile)


with open('X_test.pkl', 'rb') as picklefile:
    X_test = pickle.load(picklefile)


with open('y_test.pkl', 'rb') as picklefile:
    y_test = pickle.load(picklefile)


model = Sequential()

# conv filters of 5x5 each

# Layer 1
model.add(Convolution2D(32, (5, 5), input_shape=(input_size[0], input_size[1], 3), border_mode='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 2
model.add(Convolution2D(64, (5, 5), border_mode='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

# Layer 3
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.5))

# Layer 4

model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dropout(0.5))

# Layer 5
'''
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5))
'''

model.add(Dense(2))
model.add(Activation("softmax"))
#sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

# In[21]:


traindata = np.stack(X_train)
testdata = np.stack(X_test)
trainlabel = to_categorical(y_train)
testlabel = to_categorical(y_test)


# In[22]:


model.fit(traindata, trainlabel, batch_size=128, epochs=10, verbose=1)
print("Model training complete...")
(loss, accuracy) = model.evaluate(testdata, testlabel, batch_size = 128, verbose = 1)
print("accuracy: {:.2f}%".format(accuracy * 100))


##epoch : 1, train accuracy: .5450, test accuracy:  41.82%

## epoch: 5, train accuracy: 0.7140, test accuracy: 53.72%

## epoch: 10, train accuracy: 0.8820, test accuracy:  67.70%
# In[ ]:
print(model.summary())
