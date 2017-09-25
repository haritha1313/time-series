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
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


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

X= series.values
train, test = X[0:-12], X[-12:]

history = [x for x in train]
predictions=list()
for i in range(len(test)):
    predictions.append(history[-1])
    history.append(test[i])
    
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)
"""
plt.plot(test)
plt.plot(predictions)
plt.show()
"""
"""
supervised = timeseries_to_supervised(X,1)      #as supervised
print(supervised.head())
"""
"""
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
"""

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
