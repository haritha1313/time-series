# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 11:25:26 2017

@author: pegasus
"""
from math import sqrt
from pandas import read_csv
from pandas import concat, DataFrame
from datetime import datetime
from matplotlib import pyplot
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')

def series_to_supervised(data, n_in=1, n_out =1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df=DataFrame(data)
    cols, names = list(), list()
    
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names+=[('var%d(t-%d)'%(j+1,i)) for j in range(n_vars)]
    
    for i in range(0,n_out):
        cols.append(df.shift(-i))
        if i==0:
            names+=[('var%d(t)'%(j+1))for j in range(n_vars)]
        else:
            names+=[('vars%d(t+%d)'%(j+1, i)) for j in range(n_vars)]
        
    agg=concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg
    

"""
dataset = read_csv('raw.csv', parse_dates=[['year','month','day','hour']], index_col=0, date_parser=parse)
dataset.drop('No', axis=1,inplace=True)

dataset.columns = ['pollution','dew','temp','press','wnd_dir','wnd_spd','snow','rain']
dataset.index.name = 'date'

dataset['pollution'].fillna(0,inplace=True)

dataset=dataset[24:]
print(dataset.head(5))

dataset.to_csv('pollution.csv')
"""

dataset = read_csv('pollution.csv', header=0, index_col=0)
values=dataset.values

"""
groups=[0,1,2,3,5,6,7]
i=1

pyplot.figure()
for group in groups:
    pyplot.subplot(len(groups),1,i)
    pyplot.plot(values[:,group])
    pyplot.title(dataset.columns[group], y=0.5, loc='right')
    i+=1
pyplot.show()
"""

encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
values =  values.astype('float32')
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(values)
reframed = series_to_supervised(scaled, 1, 1)
reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
#print(reframed.head())

values = reframed.values
n_train_hours = 365*24
train = values[:n_train_hours,:]
test = values[n_train_hours:,:]
train_X, train_y = train[:,:-1], train[:,-1]
test_X, test_y = test[:,:-1], test[:,-1]

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0],1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X,test_y) , verbose=2, shuffle=False)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
inv_yhat = concatenate((yhat, test_X[:,1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

test_y = test_y.reshape((len(test_y),1))
inv_y = concatenate((test_y, test_X[:,1:]),axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)