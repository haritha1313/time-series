# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 10:46:23 2017

@author: pegasus
"""

from pandas import read_csv, datetime, DataFrame, concat
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')
    
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars=1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names=list(), list()
    
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names+=[('vars%d(t-%d)' % (j+1,i)) for j in range(n_vars)]
        
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i==0:
            names+=[('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names+=[('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    agg = concat(cols, axis=1)
    agg.columns=names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def prepare_data(series, n_test, n_lag, n_seq):
    raw_values = series.values
    raw_values = raw_values.reshape(len(raw_values),1)
    supervised = series_to_supervised(raw_values, n_lag, n_seq)
    supervised_values = supervised.values
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    return train, test

def persistence(last_ob, n_seq):
    return [last_ob for i in range(n_seq)]
    
def make_forecasts(train, test, n_lag, n_seq):
    forecasts=list()
    for i in range(len(test)):
        X, y = test[i, 0:n_lag], test[i, n_lag:]
        forecast = persistence(X[-1], n_seq)
        forecasts.append(forecast)
    return forecasts

def evaluate_forecasts(test, forecasts, n_lag, n_seq):
    for i in range(n_seq):
        actual = test[:, (n_lag+i)]
        predicted = [forecast[i] for forecast in forecasts]
        rmse = sqrt(mean_squared_error(actual, predicted))
        print('t+%d RMSE: %f' % ((i+1), rmse))
        
def plot_forecasts( series, forecasts, n_test):
    plt.plot(series.values)
    
    for i in range(len(forecasts)):
        off_s = len(series) - 12 +i -1
        off_e = off_s + len(forecasts[i]) + 1
        xaxis = [x for x in range(off_s, off_e)]
        yaxis = [series.values[off_s]] + forecasts[i]
        plt.plot(xaxis, yaxis, color='red')
    plt.show()
    
series = read_csv('shampoo-sales.csv', header = 0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
n_lag = 1
n_seq = 3
n_test = 10
train, test = prepare_data(series, n_test, n_lag, n_seq)
forecasts = make_forecasts(train, test, n_lag, n_seq)
evaluate_forecasts(test, forecasts, n_lag, n_seq)
plot_forecasts(series, forecasts, n_test+2)

"""
print(series.head())
series.plot()
plt.show()
"""