
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import glob 
import os
import sys
import scipy.io.wavfile as wav
import scipy.io
from scipy import signal
import tensorflow as tf

import random
from skimage.transform import  resize

# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#%% load training data


os.chdir(r'F:\cod_sound_analysis\2018\WA_hydrophones\tutorial')

#%% file 1
name='20180305_180000_ch1.wav'

df=pd.read_csv(name[:-8]+'.csv')
t_ann=df['tmin'] + (df['tmax']-df['tmin'])*0.5


h5f = h5py.File(os.path.basename(name)+'_sg.h5','r')
x_train_sg = h5f['x_train_sg'][:]
t1 = h5f['t1'][:]
t2 = h5f['t2'][:]
window = h5f['window'][()]
overlap = h5f['overlap'][()]
h5f.close()

score_train=np.zeros(len(t1))
for ix in range(len(t1)):      
    sumcall =   np.sum( (t_ann>=t1[ix]+window*0.3) & (t_ann<t2[ix]-window*0.3) ) 
    if sumcall>0:
        score_train[ix]=1
            
np.sum(score_train)       


# plt.figure(2)
# plt.clf()
# k=1
# for ixs in np.where(score_train==True)[0]:
#     plt.subplot(10,10,k)
#     k=k+1
#     plt.imshow( x_train_sg[ixs,:,:,0] )

# plt.tight_layout()


#%% file 2
name='20180306_001316_ch1.wav'

df=pd.read_csv(name[:-8]+'.csv')
t_ann=df['tmin'] + (df['tmax']-df['tmin'])*0.5


h5f = h5py.File(os.path.basename(name)+'_sg.h5','r')
x_train_sg_2 = h5f['x_train_sg'][:]
t1 = h5f['t1'][:]
t2 = h5f['t2'][:]
window = h5f['window'][()]
overlap = h5f['overlap'][()]
h5f.close()

score_train_2=np.zeros(len(t1))
for ix in range(len(t1)):      
    sumcall =   np.sum( (t_ann>=t1[ix]+window*0.3) & (t_ann<t2[ix]-window*0.3) ) 
    if sumcall>0:
        score_train_2[ix]=1
            
np.sum(score_train_2)       


# plt.figure(3)
# plt.clf()
# k=1
# for ixs in np.where(score_train_2==True)[0]:
#     plt.subplot(10,10,k)
#     k=k+1
#     plt.imshow( x_train_sg[ixs,:,:,0] )

# plt.tight_layout()

#%%

#%%

st =np.concatenate([score_train,score_train_2])

xt =np.concatenate([x_train_sg,x_train_sg_2],axis=0)

#%%


ix_score = np.where(st==True)[0]
ix_noscore = np.where(st==False)[0]
ix_random= random.sample( list(ix_noscore) , len(ix_score)*3 )

x_train_new=xt[ix_score,:,:,:]
x_train_new = np.concatenate([x_train_new,  xt[ix_random,:,:,:]] ,axis=0)
score_train_new =  np.concatenate([ np.ones(len(ix_score)) , np.zeros(len(ix_random)) ])

#%%


input_shape = (100, 100, 1)


# Importing the required Keras modules containing model and layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(2,activation=tf.nn.softmax)) # nr of output categories

model.summary()

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
# model.fit(x=x_test_sg,y=score_test, epochs=10)

model.fit(x=x_train_new,y=score_train_new, epochs=20)


# model.evaluate(x_train_sg, score_train)

pred = model.predict( x_train_sg )
score=score_train

score_predict= pred.argmax(axis=1)

tpr= np.sum( (score_predict==1) & (score==1)) / np.sum(score)

fpr= np.sum( (score_predict==1) & (score==0)) / np.sum( score==0 )

print('tpr '+str(tpr) + ' fpr '+str(fpr))
#%%

model.save('cod_grunt_tensorflow')

#%%
