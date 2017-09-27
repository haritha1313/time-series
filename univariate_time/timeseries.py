# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 12:59:32 2017

@author: pegasus
"""

from pandas import read_csv
from pandas import datetime
from pandas import DataFrame, concat, Series
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM


def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]

    X=X.reshape(X.shape[0], 1, X.shape[1])
    
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful = True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs =1, batch_size=batch_size, verbose=0,shuffle=False)
        model.reset_states()
    return model
    
def scale(train, test):
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler = scaler.fit(train)
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    
    test = test.reshape(test.shape[0],test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled
    
def invert_scale(scaler, X, value):
    new_row = [x for x in X]+[value]
    array = numpy.array(new_row)
    array = array.reshape(1,len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0,-1]
    
def forecast_lstm(model, batch_size, X):
    X=X.reshape(1,1,len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]
    
def timeseries_to_supervised(data, lag=1):
    df=DataFrame(data)
    columns =[df.shift(i) for i in range(1,lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0,inplace =True)
    return df

def difference(dataset, interval=1):
    diff =list()
    for i in range(interval, len(dataset)):
        value = dataset[i]-dataset[i-interval]
        diff.append(value)
    return Series(diff)
        
def inverse_difference(history, yhat, interval =1):
    return yhat+history[-interval]

def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')

series = read_csv('shampoo.csv', header=None, parse_dates=[0], index_col =0, squeeze=True, date_parser = parser)

raw_values= series.values
diff_values = difference(raw_values, 1)

supervised = timeseries_to_supervised(diff_values,1)
supervised_values = supervised.values

train, test = supervised_values[0:-12], supervised_values[-12:]

scaler, train_scaled, test_scaled = scale(train, test)

lstm_model = fit_lstm(train_scaled, 1, 3000, 4)
train_reshaped = train_scaled[:,0].reshape(len(train_scaled),1,1)
lstm_model.predict(train_reshaped, batch_size=1)

predictions= list()

for i in range(len(test_scaled)):
    X, y = test_scaled[i,0:-1], test_scaled[i,-1]
    yhat = forecast_lstm(lstm_model, 1 , X)
    
    yhat = invert_scale(scaler, X, yhat)
    predictions.append(yhat)
    expected = raw_values[len(train)+i+1]
    print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
    
rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
print('Test RMSE: %.3f' %rmse)

plt.plot(raw_values[-12:])
plt.plot(predictions)
plt.show()
"""
history = [x for x in train]
predictions=list()
for i in range(len(test)):
    predictions.append(history[-1])
    history.append(test[i])
    
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)

plt.plot(test)
plt.plot(predictions)
plt.show()

supervised = timeseries_to_supervised(X,1)      #as supervised
print(supervised.head())


#converting to stationary time series
differenced = difference(series,1)
print(differenced.head())
#inverting
inverted = list()
for i in range(len(differenced)):
    value = inverse_difference(series, differenced[i], len(series)-i)
    inverted.append(value)
inverted = Series(inverted)
print(inverted.head())


X=series.values
X=X.reshape(len(X),1)
scaler = MinMaxScaler(feature_range=(-1,1))
scaler = scaler.fit(X)
scaled_X=scaler.transform(X)
scaled_series = Series(scaled_X[:,0])
print(scaled_series.head())

inverted_X = scaler.inverse_transform(scaled_X)
inverted_series = Series(inverted_X[:,0])
print(inverted_series.head())
"""