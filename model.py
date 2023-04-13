# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 13:38:44 2023

@author: caisa
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import numpy as np

#importera det transponerade bladet i excelarket, exkludera första kolumnen
dataset1 = pd.read_excel(r"C:\Users\caisa\Documents\Fysik\VT23\EXJOBB\modell\model_data.xlsx", sheet_name='Blad2', usecols='B:U')
df_lunga = pd.get_dummies(dataset1['tumour_pos (0=upper, 1=lower, 2=middle)'], prefix='lunga') #one hot encoding
dataset = pd.concat([dataset1, df_lunga], axis=1)
dataset = dataset.drop(columns='tumour_pos (0=upper, 1=lower, 2=middle)')


#%%
"""
modell v4 - lägg till modell som gör binär klassificering utefter första modellen
"""
# x-dataset är datan vi använder
x = dataset.drop(columns=['tumour_CT0', 'tumour_CT13', 'tumour_CT25', 'tumour_CT38', 'tumour_CT50',
                          'tumour_CT63', 'tumour_CT75', 'tumour_CT88'])
#y-dataset är datan vi vill förutspå
y = pd.DataFrame(dataset, columns=['tumour_CT0', 'tumour_CT13', 'tumour_CT25', 'tumour_CT38', 'tumour_CT50', 
                                   'tumour_CT63', 'tumour_CT75', 'tumour_CT88'])

#split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)  #vi sätter 20% av datan åt sidan för endast test för att undvika overfitting

#%%
#Modell ett - do teh fitting
#build and train model
model1 = tf.keras.models.Sequential()
reg_l1 = tf.keras.regularizers.L1(l1=0.0001)  #regulizer
reg_l2 = tf.keras.regularizers.L2(l2=0.0001)
#lägger till lager av NN
model1.add(tf.keras.layers.Dense(128, input_shape=(x_train.shape[1],), activation='relu', kernel_regularizer=reg_l1))
model1.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=reg_l1))
model1.add(tf.keras.layers.Dense(8, activation='linear')) #Output layer  #vill att output ska vara åtta punkter, dvs puntker i de åtta CT faserna
model1.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.MeanAbsoluteError(), 'acc'])  #mse + accuracy på metrics - vilka siffror vi vill se
epoch=400
c_model = model1.fit(x_train, y_train, epochs=epoch, validation_data=(x_test, y_test))

#%%
"""
Försök två på del två - istället för att ha ett ANN för den klassificeringen, så kan vi använda threshold och bara kolla hur
många gånger det bli false och ha det som någon typ av accuracy?
"""

#dela in i hög, medel och låg korrelationspatienter
high_corr_x = x.query('[0,1,2,3,5,6,7,9,10]')  #query function = drop all except indices
high_corr_y = y.query('[0,1,2,3,5,6,7,9,10]')
mid_corr_x = x.query('[6,8,18,29,30,32,34,39,41]')
mid_corr_y = y.query('[6,8,18,29,30,32,34,39,41]')
low_corr_x = x.query('[44, 23, 13]')
low_corr_y = y.query('[44, 23, 13]')

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
    
    #kollar om skillnaden mellan predicted value och ground truth är större än MSE mm
    y_pred_threshold =np.abs(np.array(prediction) - np.array(true_val))<= threshold
    nofalse = np.count_nonzero(y_pred_threshold==False)
    notrue = np.count_nonzero(y_pred_threshold==True)
    perc_true = round((notrue/(nofalse+notrue))*100, 1)
    
    result = [threshold, perc_true]
    return result

#få model predictions
thresh = threshold(model1.predict(x_test), y_test)
high_thresh = threshold(model1.predict(high_corr_x), high_corr_y)
mid_thresh = threshold(model1.predict(mid_corr_x), mid_corr_y)
low_thresh = threshold(model1.predict(low_corr_x), low_corr_y)

print(f'sqrt MSE total: {round(thresh[0], 1)} mm \nsqrt MSE högkorr-pat: {round(high_thresh[0],1)} mm \nsqrt MSE medelkorr-pat: {round(mid_thresh[0],1)} mm \nsqrt MSE lågkorr-pat: {round(low_thresh[0],1)} mm \n')
print(f'perc true total: {thresh[1]}% \nperc true high: {high_thresh[1]}% \nperc true mid: {mid_thresh[1]}% \nperc true low: {low_thresh[1]}%')

#%%
#plotta träningen i model1
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
#Plotta predicitions mot facit
# get the model predictions on the test data
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


#plotta de andra patienterna
rest_y_pred = y_pred[2:]
rest_y_test = y_test[2:]

#dela in i två figurer - 2x2, 2x2
figb, axb = plt.subplots(2,2)
figb.set_size_inches(8,8, forward=True)
for i, ax in enumerate(axb.flat):
    #legend_element = [plt.Line2D([0], [0],  label=f'Test patient {i}')]
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
    #legend_element = [plt.Line2D([0], [0],  label=f'Test patient {i}')]
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

#%%
"""
#plotta alla pat?
y_pred = model1.predict(x_test)

fig, axs = plt.subplots(3,3)
fig.set_size_inches(10,10)
for i, ax in enumerate(axs.flat):
    print(f'picking {i}th test')
    p = y_pred[i]
    r = y_test.iloc[i]
    upper_p, lower_p = (p+threshold), (p-threshold)
    a = ax.plot(ct_phase, p, label='predicted value')
    b = ax.plot(ct_phase, r, label='actual value')
    ax.fill_between(ct_phase, upper_p, lower_p, alpha=0.2, color='blue', label='$\pm \sqrt{MSE}$')
    ax.set_ylim(-4, 14)
    ax.set_yticks(np.arange(-2,14, 2))
    ax.set_ylabel('Tumour motion mm')
    ax.grid()
    ax.legend()  
    ax.text(min(ct_phase), ax.get_ylim()[1]-1,f'Test patient {i+1}', bbox={'facecolor':'white', 'alpha':0.5, 'pad':5})
    ax.set_xlabel('CT phase')
plt.tight_layout()
"""