#!/usr/bin/env python
# coding: utf-8

# ## LSTM network for multi-step time series forecasting.

# In[ ]:


# load and plot dataset
import glob
import pandas as pd
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from numpy import array

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


###############################################################################
# read input omniture .csv files into dataframe
###############################################################################
import glob
import pandas as pd

# omniture files path
path = r'C:\Users\Raja Harsha\Documents\DE\LSTM\'

all_files = glob.glob(path + "/*.csv")
df_omniture = pd.DataFrame()
list_ = []
for f in all_files:
    df = pd.read_csv(f, index_col=None, header=0)
    list_.append(df)

df_omniture = pd.concat(list_)
df_omniture.head()

df_omniture.columns = ['zip', 'pv', 'date']

###############################################################################
# filter and calculate monthly page views for zip code data: XXXXX
###############################################################################

df_zipdata = df_omniture[df_omniture['zip'] == 'XXXXX']

# Convert that column into a datetime datatype
df_zipdata['Month'] = pd.to_datetime(df_zipdata['date'])

# Set the datetime column as the index
df_zipdata.index = df_zipdata['Month'] 

# Drop date, zip columns 
df_zipdata = df_zipdata.drop(['Month','zip'], 1)

# aggregate the daily data to monthly
df_m_zipdata = df_zipdata.resample('M').sum()

df_m_zipdata.head()


# In[ ]:


df_m_zipdata.tail()


# In[ ]:


series = df_m_zipdata


# In[ ]:


# line plot
series.plot()
pyplot.show()


# ### Multi-Step Forecast: Every 3 Months

# """
# Dec,	Jan, Feb, Mar
# Jan,	Feb, Mar, Apr
# Feb,	Mar, Apr, May
# Mar,	Apr, May, Jun
# Apr, 	May, Jun, Jul
# May,	Jun, Jul, Aug
# Jun,	Jul, Aug, Sep
# Jul,	Aug, Sep, Oct
# Aug,	Sep, Oct, Nov
# Sep,	Oct, Nov, Dec
# """

# ### Time Series for Supervised Learning Problem

# In[ ]:


# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
    
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
            
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
    
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


# ### Transform Series into Test and Train Datasets

# In[ ]:


# transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq):
	# extract raw values
	raw_values = series.values
	raw_values = raw_values.reshape(len(raw_values), 1)
    
	# transform into supervised learning problem X, y
	supervised = series_to_supervised(raw_values, n_lag, n_seq)
	supervised_values = supervised.values
    
	# split into train and test sets
	train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    
	return train, test


# In[ ]:


# Apply Time Series Conversion and derive Test-Train Datasets here.

# configure
n_lag=1
n_seq=3
n_test=5

# prepare data
train, test = prepare_data(series, n_test, n_lag, n_seq)
print(test)
print('\nDatasets Shape:')
print('Train: %s, Test: %s' % (train.shape, test.shape))


# In[ ]:


train


# In[ ]:


test


# ### Make Persistence Forecasts

# In[ ]:


# make a persistence forecast
def persistence(last_ob, n_seq):
	return [last_ob for i in range(n_seq)]


# In[ ]:


# evaluate the persistence model
def make_forecasts(train, test, n_lag, n_seq):
	forecasts = list()
	for i in range(len(test)):
		X, y = test[i, 0:n_lag], test[i, n_lag:]
		# make forecast
		forecast = persistence(X[-1], n_seq)
		# store the forecast
		forecasts.append(forecast)
	return forecasts


# In[ ]:


# call forecasting function
forecasts = make_forecasts(train, test, 1, 3)


# In[ ]:


forecasts


# ### Evaluate Forecasts
# 
# Calculate RMSE for each time step of multi-step forecast, this gives us 3 RMSE scores.

# In[ ]:


# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, n_lag, n_seq):
	for i in range(n_seq):
		actual = test[:,(n_lag+i)]
		predicted = [forecast[i] for forecast in forecasts]
		rmse = sqrt(mean_squared_error(actual, predicted))
		print('t+%d RMSE: %f' % ((i+1), rmse))


# In[ ]:


evaluate_forecasts(test, forecasts, 1, 3)


# ### Plotting Forecasts

# In[ ]:


# plot the forecasts in the context of the original dataset
def plot_forecasts(series, forecasts, n_test):
	# plot the entire dataset in blue
	pyplot.plot(series.values)
	# plot the forecasts in red
	for i in range(len(forecasts)):
		off_s = len(series) - 7 + i - 1
		off_e = off_s + len(forecasts[i]) + 1
		xaxis = [x for x in range(off_s, off_e)]
		yaxis = [series.values[off_s]] + forecasts[i]
		pyplot.plot(xaxis, yaxis, color='red')
	# show the plot
	pyplot.show()


# In[ ]:


# plot forecasts
plot_forecasts(series, forecasts, 7)


# Line plot of sales dataset with multi-step persistence forecasts. <br>
# The context shows how naive the persistence forecasts actually are.

# ## Multi-Step LSTM Network

# ### Data Preparation: Stationarizing and Scaling

# In[ ]:


# stationarizing
# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)


# In[ ]:


# scaling
# we can use the MinMaxScaler from the sklearn library to scale the data.

# transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq):
	# extract raw values
	raw_values = series.values
	# transform data to be stationary
	diff_series = difference(raw_values, 1)
	diff_values = diff_series.values
	diff_values = diff_values.reshape(len(diff_values), 1)
	# rescale values to -1, 1
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaled_values = scaler.fit_transform(diff_values)
	scaled_values = scaled_values.reshape(len(scaled_values), 1)
	# transform into supervised learning problem X, y
	supervised = series_to_supervised(scaled_values, n_lag, n_seq)
	supervised_values = supervised.values
	# split into train and test sets
	train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
	return scaler, train, test


# In[ ]:


# prepare data
scaler, train, test = prepare_data(series, n_test, n_lag, n_seq)


# In[ ]:


test


# ### Fit LSTM Network

# In[ ]:


# fit an LSTM network to training data
def fit_lstm(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
	# reshape training into [samples, timesteps, features]
	X, y = train[:, 0:n_lag], train[:, n_lag:]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	# design network
	model = Sequential()
	model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(y.shape[1]))
	model.compile(loss='mean_squared_error', optimizer='adam')
	# fit network
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
		model.reset_states()
	return model


# In[ ]:


# fit model
model = fit_lstm(train, 1, 3, 1, 1000, 1)


# ### Make LSTM Forecasts

# In[ ]:


# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch):
	# reshape input pattern to [samples, timesteps, features]
	X = X.reshape(1, 1, len(X))
	# make forecast
	forecast = model.predict(X, batch_size=n_batch)
	# convert to array
	return [x for x in forecast[0, :]]


# In[ ]:


# evaluate the persistence model
def make_forecasts(model, n_batch, train, test, n_lag, n_seq):
	forecasts = list()
	for i in range(len(test)):
		X, y = test[i, 0:n_lag], test[i, n_lag:]
		# make forecast
		forecast = forecast_lstm(model, X, n_batch)
		# store the forecast
		forecasts.append(forecast)
	return forecasts


# In[ ]:


# make forecasts
forecasts = make_forecasts(model, 1, train, test, 1, 3)


# ### Invert Transforms

# In[ ]:


# invert differenced forecast
def inverse_difference(last_ob, forecast):
	# invert first forecast
	inverted = list()
	inverted.append(forecast[0] + last_ob)
	# propagate difference forecast using inverted first value
	for i in range(1, len(forecast)):
		inverted.append(forecast[i] + inverted[i-1])
	return inverted


# In[ ]:


# inverse data transform on forecasts
def inverse_transform(series, forecasts, scaler, n_test):
	inverted = list()
	for i in range(len(forecasts)):
		# create array from forecast
		forecast = array(forecasts[i])
		forecast = forecast.reshape(1, len(forecast))
		# invert scaling
		inv_scale = scaler.inverse_transform(forecast)
		inv_scale = inv_scale[0, :]
		# invert differencing
		index = len(series) - n_test + i - 1
		last_ob = series.values[index]
		inv_diff = inverse_difference(last_ob, inv_scale)
		# store
		inverted.append(inv_diff)
	return inverted


# In[ ]:


# inverse transform forecasts and test
forecasts = inverse_transform(series, forecasts, scaler, n_test+2)


# In[ ]:


# calculate RMSE values
actual = [row[n_lag:] for row in test]
actual = inverse_transform(series, actual, scaler, n_test+2)


# In[ ]:


def evaluate_forecasts(test, forecasts, n_lag, n_seq):
	for i in range(n_seq):
		actual = [row[i] for row in test]
		predicted = [forecast[i] for forecast in forecasts]
		rmse = sqrt(mean_squared_error(actual, predicted))
		print('t+%d RMSE: %f' % ((i+1), rmse))


# In[ ]:


# evaluate forecasts
evaluate_forecasts(actual, forecasts, n_lag, n_seq)


# In[ ]:


# plot forecasts
plot_forecasts(series, forecasts, n_test+2)


# In[ ]:


actual


# In[ ]:


forecasts


# In[ ]:


abs_error = []
for i in [0,1,2,3,4]:
    for j in [0]:
        diff = (actual[i][j][0] - forecasts[i][j][0])*100/(actual[i][j][0])
        abs_error.append(abs(diff))
        print (diff)

# absolute mean error
import numpy as np
np.mean(abs_error)

