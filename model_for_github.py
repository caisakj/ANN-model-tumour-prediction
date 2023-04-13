# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 11:39:19 2023

@author: caisa
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import numpy as np

#Import the correct sheet and columns in file
dataset1 = pd.read_excel(r"C:\Users\caisa\Documents\Fysik\VT23\EXJOBB\modell\model_data.xlsx", sheet_name='Blad2', usecols='B:U')
df_lunga = pd.get_dummies(dataset1['tumour_pos (0=upper, 1=lower, 2=middle)'], prefix='lunga') #one hot encoding
dataset = pd.concat([dataset1, df_lunga], axis=1)
dataset = dataset.drop(columns='tumour_pos (0=upper, 1=lower, 2=middle)')

# x data set is input
x = dataset.drop(columns=['tumour_CT0', 'tumour_CT13', 'tumour_CT25', 'tumour_CT38', 'tumour_CT50',
                          'tumour_CT63', 'tumour_CT75', 'tumour_CT88'])
#y data set is output/the data we want to predict
y = pd.DataFrame(dataset, columns=['tumour_CT0', 'tumour_CT13', 'tumour_CT25', 'tumour_CT38', 'tumour_CT50', 
                                   'tumour_CT63', 'tumour_CT75', 'tumour_CT88'])

#split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)  #vi sätter 20% av datan åt sidan för endast test för att undvika overfitting

#%%
#build and train model
model1 = tf.keras.models.Sequential()
reg_l1 = tf.keras.regularizers.L1(l1=0.0001)  #regulariser
model1.add(tf.keras.layers.Dense(128, input_shape=(x_train.shape[1],), activation='relu', kernel_regularizer=reg_l1))
model1.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=reg_l1))
model1.add(tf.keras.layers.Dense(8, activation='linear')) #Output layer, shape of 8 becuase of the eight phases the breathing cycle is divided into
model1.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.MeanAbsoluteError(), 'acc'])  #which parameters we want to extract
epoch=400
c_model = model1.fit(x_train, y_train, epochs=epoch, validation_data=(x_test, y_test))

#%%
#Creating and investigating the MSE interval

def threshold(prediction, true_val):
    """
    Parameters
    ----------
    prediction : predicted values from model.
    true_val : true values that the predictions are compared to. has to have same size as prediction

    Returns
    -------
    result : list with threshold value sqrt(MSE) and percentage of true values that are within the estimated threshold

    """
    mse_model = mean_squared_error(true_val, prediction) 
    threshold = np.sqrt(mse_model) 
    
    #check if difference between predicted value and ground truth is larger than threshold
    y_pred_threshold =np.abs(np.array(prediction) - np.array(true_val))<= threshold
    nofalse = np.count_nonzero(y_pred_threshold==False)
    notrue = np.count_nonzero(y_pred_threshold==True)
    perc_true = round((notrue/(nofalse+notrue))*100, 1)
    
    result = [threshold, perc_true]
    return result

#get model predictions
thresh = threshold(model1.predict(x_test), y_test)
print(f'sqrt MSE total: {round(thresh[0], 1)} mm')
print(f'perc true total: {thresh[1]}%')

#%%
#plot training of model
train_loss = c_model.history['loss']
val_loss = c_model.history['val_loss']
acc = c_model.history['acc']
val_acc = c_model.history['val_acc']
arb_x = np.linspace(1,epoch,epoch)

fig, ax = plt.subplots(1, 2)
fig.set_size_inches(8,4)
ax[0].plot(arb_x, train_loss, label='Train loss')
ax[0].plot(arb_x, val_loss, label='Validation loss')
ax[0].set_title('Model Loss')
ax[0].set_yticks(np.arange(0, max(train_loss), 2.5))
ax[1].plot(arb_x, acc, label='Train accuracy')
ax[1].set_title('Model Accuracy')
ax[1].set_yticks(np.arange(0,max(acc),0.05))

for ax in ax.flat:
    ax.legend()
    ax.grid()
    ax.set_xticks(np.arange(0,epoch,100))
plt.tight_layout()  

#%%
#Plot predictions against true values and MSE interval

y_pred = model1.predict(x_test)

# plot the four first predictions and actual values for each target variable
p1, p2= y_pred[0], y_pred[1]
r1, r2= y_test.iloc[0], y_test.iloc[1]
ct_phase=[0,13,25,38,50,63,75,88]
threshold = thresh[0]

fig, axs = plt.subplots(1,2)
fig.set_size_inches(8,4)
for i, ax in enumerate(axs.flat):
    p = y_pred[i]
    r = y_test.iloc[i]
    upper_p, lower_p = (p+threshold), (p-threshold)
    a = ax.plot(ct_phase, p, 'o-', label='predicted value')
    b = ax.plot(ct_phase, r, 'v-', label='actual value')
    ax.fill_between(ct_phase, upper_p, lower_p, alpha=0.2, color='blue', label='$\pm \sqrt{MSE}$')
    ax.set_ylim(-4, 14)
    ax.set_yticks(np.arange(-2,14, 2))
    ax.set_ylabel('Tumour motion mm')
    ax.grid()
    ax.legend()  
    ax.text(min(ct_phase), ax.get_ylim()[1]-1,f'Test patient {i+1}', bbox={'facecolor':'white', 'alpha':0.5, 'pad':5})
    ax.set_xlabel('CT phase')
plt.tight_layout()


#plot the rest of the test patients
rest_y_pred = y_pred[2:]
rest_y_test = y_test[2:]

#divide into two figures - 2x2, 2x2
figb, axb = plt.subplots(2,2)
figb.set_size_inches(8,8, forward=True)
for i, ax in enumerate(axb.flat):
    p = rest_y_pred[i]
    r = rest_y_test.iloc[i]
    upper_p, lower_p = (p+threshold), (p-threshold)
    a = ax.plot(ct_phase, p, 'o-', label='predicted value')
    b = ax.plot(ct_phase, r, 'v-', label='actual value')
    ax.fill_between(ct_phase, upper_p, lower_p, alpha=0.2, color='blue', label='$\pm \sqrt{MSE}$')
    ax.set_ylim(-4, 14)
    ax.set_yticks(np.arange(-2,14, 2))
    ax.set_ylabel('Tumour motion mm')
    ax.grid()
    ax.legend()  
    ax.text(min(ct_phase), ax.get_ylim()[1]-1,f'Test patient {i+3}', bbox={'facecolor':'white', 'alpha':0.5, 'pad':5})
    ax.set_xlabel('CT phase')
plt.tight_layout()


rest_y_pred = y_pred[6:]
rest_y_test = y_test[6:]
figc = plt.figure()
gs = plt.GridSpec(2, 2)
axc = [figc.add_subplot(gs[0, 0]), figc.add_subplot(gs[0,1]), figc.add_subplot(gs[1,0])]
figc.set_size_inches(8,8, forward=True)
for i, ax in enumerate(axc):
    p = rest_y_pred[i]
    r = rest_y_test.iloc[i]
    upper_p, lower_p = (p+threshold), (p-threshold)
    a = ax.plot(ct_phase, p, 'o-', label='predicted value')
    b = ax.plot(ct_phase, r, 'v-', label='actual value')
    ax.fill_between(ct_phase, upper_p, lower_p, alpha=0.2, color='blue', label='$\pm \sqrt{MSE}$')
    ax.set_ylim(-4, 14)
    ax.set_yticks(np.arange(-2,14, 2))
    ax.set_ylabel('Tumour motion mm')
    ax.grid()
    ax.legend()  
    ax.text(min(ct_phase), ax.get_ylim()[1]-1,f'Test patient {i+7}', bbox={'facecolor':'white', 'alpha':0.5, 'pad':5})
    ax.set_xlabel('CT phase')
plt.tight_layout()
