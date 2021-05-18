# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 18:10:44 2017
@author: Raja Harsha
"""

import glob
import pandas as pd

###############################################################################
# read input omniture .csv files into dataframe
###############################################################################

# omniture files path
path = r'C:\Users\Raja Harsha\Documents\DE\Semster Work\Homes_LSTM\omniture'

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
# filter and calculate monthly page views for zip code data: Maryland 21061
###############################################################################

df_zipdata = df_omniture[df_omniture['zip'] == '21061']

# Convert that column into a datetime datatype
df_zipdata['Month'] = pd.to_datetime(df_zipdata['date'])

# Set the datetime column as the index
df_zipdata.index = df_zipdata['Month'] 

# Drop date, zip columns 
df_zipdata = df_zipdata.drop(['Month','zip'], 1)

# aggregate the daily data to monthly
df_m_zipdata = df_zipdata.resample('M').sum()

df_m_zipdata.head()

    



