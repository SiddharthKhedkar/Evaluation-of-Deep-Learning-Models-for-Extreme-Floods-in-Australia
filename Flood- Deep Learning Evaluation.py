#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import folium
import os
import glob
import random


import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.distributions.empirical_distribution import ECDF

from pmdarima.arima import auto_arima

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from numpy import array
import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.layers import Reshape
from keras.layers import Input
from keras.layers import Dropout
from keras.layers import Concatenate
from keras.layers import RepeatVector

from tensorflow.keras.optimizers import SGD
from keras.preprocessing.sequence import TimeseriesGenerator

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers import Conv1D
from keras.layers import MaxPooling1D

from keras.layers import TimeDistributed, RepeatVector
from tensorflow.keras import layers

import tensorflow as tf
from datetime import datetime
from textwrap import wrap
import itertools

from sklearn.model_selection import train_test_split


# In[2]:


os.chdir(os.path.dirname(os.path.abspath("__file__")))


# In[3]:


timeseries_dfs = []
summary_dfs = []

# Read all csv files from directory
# Sort files into timeseries and summary data
for file_path in glob.glob('D:/Flood Deep Learning/**/*.csv', recursive=True):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    df = pd.read_csv(file_path, low_memory=False) 

    #skip these files
    if file_name in ['streamflow_QualityCodes']:
        continue

    if 'year' in df.columns:    
        df['source'] = file_name
        df= df[df['year'] > 1990]
        df= df.drop_duplicates(['year','month','day'])
        timeseries_dfs.append(df)
    else:
        df = df.rename({'ID':'station_id'}, axis=1)
        df = df.set_index('station_id')
        summary_dfs.append(df)

timeseries_data = pd.concat(timeseries_dfs, axis=0, ignore_index=True)
timeseries_data['date'] = pd.to_datetime(timeseries_data[['year', 'month', 'day']])
timeseries_data = timeseries_data.drop(['year', 'month', 'day'], axis=1)

summary_data = pd.concat(summary_dfs, axis=1)


# In[4]:


class PrepareData():
    def __init__(self, timeseries_data=timeseries_data, summary_data=summary_data):
        ### Data Cleaning
        self.timeseries_data = timeseries_data.replace(-99.99,np.NaN)
        
        ### Feature Engineering
        # get precipitation deficit
        actualTransEvap_data = self.timeseries_data[self.timeseries_data['source'] == 'et_morton_actual_SILO'].drop(['source'], axis=1)
        precipitation_data = self.timeseries_data[self.timeseries_data['source'] == 'precipitation_AWAP'].drop(['source'], axis=1)
         
        actualTransEvap_data = actualTransEvap_data[actualTransEvap_data['date'].isin(precipitation_data['date'])].reset_index(drop=True)
        precipitation_data = precipitation_data[precipitation_data['date'].isin(actualTransEvap_data['date'])].reset_index(drop=True)
        
        self.precipitation_deficit = precipitation_data.drop(['date'], axis=1).subtract(actualTransEvap_data.drop(['date'], axis=1))
        self.precipitation_deficit['source'] = 'precipitation_deficit'
        self.precipitation_deficit['date'] = precipitation_data['date']
        
        # get flood probabilities
        self.streamflow_data = self.timeseries_data[timeseries_data['source'] == 'streamflow_MLd_inclInfilled'].drop(['source'], axis=1)
        self.streamflow_data = self.streamflow_data.set_index('date')
        
        self.flood_probabilities = self.streamflow_data.apply(self.flood_extent, axis=0)
        self.flood_probabilities['source'] = 'flood_probabilities'
        self.flood_probabilities['date'] = self.streamflow_data.index
        
        self.flood_indicator = self.flood_probabilities.applymap(lambda x: int(x <0.05) if pd.isnull(x) == False and isinstance(x, float) else x)
        self.flood_indicator['source'] = 'flood_indicator'
        self.flood_indicator['date'] = self.flood_probabilities['date']        
        
        # turn date into sin and cos function 
        date_min = np.min(self.flood_probabilities['date'])
        year_seconds = 365.2425*24*60*60
        year_sin = self.flood_probabilities['date'].apply(lambda x: np.sin((x-date_min).total_seconds() * (2 * np.pi / year_seconds)))
        year_cos = self.flood_probabilities['date'].apply(lambda x: np.cos((x-date_min).total_seconds() * (2 * np.pi / year_seconds)))
        all_stations = list(self.flood_probabilities.drop(columns=['source', 'date'], axis=1).columns) 
        
        df_sin = []     
        for value in year_sin:
            df_sin.append({k:value for k in all_stations})
            
        df_sin = pd.DataFrame(df_sin)
        df_sin['source'] = 'year_sin'
        df_sin['date'] = self.flood_probabilities['date']
 
        df_cos = []
        for value in year_cos:
            df_cos.append({k:value for k in all_stations})
            
        df_cos = pd.DataFrame(df_cos)
        df_cos['source'] = 'year_cos'
        df_cos['date'] = self.flood_probabilities['date']
            
        ### Return
        self.timeseries_data = pd.concat([self.timeseries_data, self.precipitation_deficit, self.flood_probabilities, df_sin, df_cos, self.flood_indicator], axis=0).reset_index(drop=True)
        self.summary_data = summary_data
        
    def get_timeseries_data(self, source, stations):      
        # filter by source
        self.data_filtered = self.timeseries_data[self.timeseries_data['source'].isin(source)]
        # pivot data by station
        self.data_filtered = self.data_filtered[['date', 'source'] + stations].pivot(index='date', columns='source', values=stations)
        # get rows with no nan
        self.data_filtered = self.data_filtered[~self.data_filtered.isnull().any(axis=1)]
        
        return self.data_filtered
        
        
    def get_data(self, source, stations):
        summary_source = [i for i in source if i in list(self.summary_data.columns)]
        timeseries_source = [i for i in source if i not in list(self.summary_data.columns)]
     
        # filter by source
        self.data_filtered = self.timeseries_data[self.timeseries_data['source'].isin(timeseries_source)]
        # pivot data by station
        self.data_filtered = self.data_filtered[['date', 'source'] + stations].pivot(index='date', columns='source', values=stations)
        # get rows with no nan
        self.data_filtered = self.data_filtered[~self.data_filtered.isnull().any(axis=1)]
        
        for station in stations:
            for variable in summary_source:
                value = self.summary_data.loc[station][variable]
                self.data_filtered[station, variable] = value
        
        return self.data_filtered.sort_index(axis=1)
    

    
    def get_train_val_test(self, source, stations, 
                           scaled=True, target=['streamflow_MLd_inclInfilled'],
                           start=None, end=None,
                           discard=0.05, train=0.6, test=0.4):
        assert 0<=discard<=1
        assert (train + test) == 1
     
        summary_source = [i for i in source if i in list(self.summary_data.columns)]
        timeseries_source = [i for i in source if i not in list(self.summary_data.columns)]        
        
        all_data = self.get_timeseries_data(timeseries_source, stations).loc[start:end]
        n_rows_all = len(all_data)
        
        all_data_discarded = all_data.iloc[int(n_rows_all*discard):]
        n_rows_discarded = len(all_data_discarded)
        
        train_df = all_data_discarded[:int(n_rows_discarded*train)]
        test_df = all_data_discarded[-int(n_rows_discarded*(test)):]
        
        if scaled == True:
            scaler = MinMaxScaler()
            scaler.fit(train_df)
            
            scaler_test = MinMaxScaler()
            scaler_test.fit(test_df)
            
            train_df = pd.DataFrame(scaler.transform(train_df), index=train_df.index, columns=train_df.columns)
            test_df = pd.DataFrame(scaler_test.transform(test_df), index=test_df.index, columns=test_df.columns)
            
     
        for station in stations:
            for variable in summary_source:
                value = self.summary_data.loc[station][variable]
                
                train_df[station, variable] = value                
                test_df[station, variable] = value 
                                  
        return train_df.sort_index(axis=1), test_df.sort_index(axis=1) 
    
    def flood_extent(self, streamflow_ts):
        station_name = streamflow_ts.name

        flow_data = pd.DataFrame(streamflow_ts)  
        na_values = flow_data[flow_data[station_name].isna()][station_name]

        flow_data = flow_data.dropna().sort_values(by=station_name, ascending=False).reset_index()
        flow_data['probability'] = (flow_data.index + 1)/(1+len(flow_data)) 
        flow_data = flow_data.sort_values(by='date').drop(['date', station_name], axis=1)['probability']
        flow_data = pd.concat([na_values, flow_data]).reset_index(drop=True) 
        flow_data.name = station_name  

        return flow_data 


# In[5]:


camels_data = PrepareData()


# In[6]:


class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 train_df, test_df, 
                 station, filtered=None, label_columns=None):

        # Store the raw data.
        self.train_df = train_df[station]
        self.test_df = test_df[station]
        self.station = station
        self.filtered = filtered
        
        # validation
        if self.filtered == 'upper_soil_filter':
            assert('upper_soil_indicator' in list(self.train_df.columns))
        
        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                        enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(self.train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data, filtered=None, shuffle=True):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
                data=data,
                targets=None,
                sequence_length=self.total_window_size,
                sequence_stride=1,
                shuffle=shuffle,
                batch_size=32,)
        
        ds = ds.map(self.split_window)
        
        if filtered == 'upper_soil_filter':
            indicator_index = list(self.train_df.columns).index('upper_soil_indicator')
            ds = ds.unbatch().filter(lambda x, y: tf.math.reduce_sum(x[:, indicator_index]) > 0).batch(32)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df, filtered=self.filtered)

    @property
    def test(self):
        return self.make_dataset(self.test_df, shuffle=False)
           
    @property
    def test_windows(self):
        total_size = self.test_array.shape[0]
        convolution_size = self.input_width
        prediction_size = self.label_width
        
        a = []
        
        for i in range(convolution_size, (total_size-prediction_size+1)):
            index_list = list(range(i, i+prediction_size))
            a.append(self.test_array[index_list])
        
        return np.squeeze(array(a), axis=2)
    
    @property
    def test_array(self):
        return array(self.test_df[self.label_columns])
       
    def test_indicator(self, filtered):
        if filtered == 'upper_soil_filter':
            return array(self.test_df['upper_soil_indicator'])       

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
            
        return result
       
    @property
    def num_features(self):
        return self.train_df.shape[1]
    
    def test_example(self, index):
        return np.array(self.test_df.iloc[index]).reshape(1, 1, self.num_features)

    def plot(self, model=None, plot_col='streamflow_MLd_inclInfilled', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                              marker='X', edgecolors='k', label='Predictions',
                              c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')
    
    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def station_percentile(self, cut, variable='streamflow_MLd_inclInfilled', station=None):
        if station==None:
            return np.percentile(self.test_df[variable], 100-cut)
        else:
            return np.percentile(self.test_df[station][variable], 100-cut)


# In[7]:


class MultiWindow():
    def __init__(self, input_width, label_width, shift,
                 train_df,test_df, stations, 
                 statics='separate', filtered=-1000, label_columns=None):
     
        # Store the raw data.
        self.train_df = train_df       
        self.test_df = test_df
        self.stations = stations
        self.total_stations = len(stations) 

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift
        self.label_columns = label_columns
        
        self.filtered = filtered
           
        self.trains = []
        self.tests = []        

        for i, s in enumerate(stations):
            window = WindowGenerator(input_width=input_width,
                                     label_width=label_width,
                                     shift=shift,
                                     train_df=train_df,
                                     test_df=test_df,
                                     station=s,
                                     label_columns=label_columns)

            padding = np.zeros((input_width, self.total_stations), dtype=np.float32)
            padding[:, i] = 1

            self.trains.append(window.train.unbatch().map(lambda x, y: (tf.concat([x, tf.convert_to_tensor(padding)], axis=1),y)))
            self.tests.append(window.test.unbatch().map(lambda x, y: (tf.concat([x, tf.convert_to_tensor(padding)], axis=1),y)))
            
    @property
    def train(self):
        ds = tf.data.Dataset.from_tensor_slices(self.trains)
        concat_ds = ds.interleave(lambda x: x, cycle_length=1, num_parallel_calls=tf.data.AUTOTUNE).filter(lambda x, y: tf.math.reduce_max(y) > self.filtered)
        
        return concat_ds.batch(32)


    @property
    def test(self):
        ds = tf.data.Dataset.from_tensor_slices(self.tests)
        concat_ds = ds.interleave(lambda x: x, cycle_length=1, num_parallel_calls=tf.data.AUTOTUNE)
        
        return concat_ds.batch(32)


    def test_windows(self, station=None):
        total_size = self.test_array(station).shape[0]
        convolution_size = self.input_width
        prediction_size = self.label_width
        
        a = []
        
        for i in range(convolution_size, (total_size-prediction_size+1)):
            index_list = list(range(i, i+prediction_size))
            a.append(self.test_array(station)[index_list])
        
        return np.squeeze(array(a), axis=2)
    
    def test_array(self, station):
        return array(self.test_df[station][self.label_columns])
    
    def station_percentile(self, cut, variable='streamflow_MLd_inclInfilled', station=None):
        if station==None:
            return np.percentile(self.test_df[variable], 100-cut)
        else:
            return np.percentile(self.test_df[station][variable], 100-cut)


# In[8]:


class MultiNumpyWindow():
    def __init__(self, input_width, label_width, shift,
                 timeseries_source, summary_source, summary_data,
                 stations, label_columns=None):
        
        train_df, test_df = camels_data.get_train_val_test(source=timeseries_source,
                                                                   stations=stations)      
        # Store the raw data.
        self.train_df = train_df
        self.test_df = test_df
        self.stations = stations
        self.total_stations = len(stations)
        
        self.num_timeseries_features = len(timeseries_source)
        self.num_static_features = len(summary_source)     
        self.timeseries_source = timeseries_source
        self.summary_source = summary_source      
    
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift
        self.label_columns = label_columns
        
        self.train_timeseries = []
        self.test_timeseries = []  
        
        self.train_static = []
        self.test_static = [] 
        
        self.train_y = []
        self.test_y = []
        
        for i, s in enumerate(stations):
            # process timeseries
            window = WindowGenerator(input_width=input_width,
                                     label_width=label_width,
                                     shift=shift,
                                     train_df=train_df,
                                     test_df=test_df,
                                     station=s,
                                     label_columns=label_columns)
            
            x_train, y_train = self.mapdata_tonumpy(window.train)
            x_test, y_test = self.mapdata_tonumpy(window.test)
            
            self.train_timeseries.extend(x_train) 
            self.test_timeseries.extend(x_test) 
            
            self.train_y.extend(y_train) 
            self.test_y.extend(y_test)      
            
            # process static
            static = summary_data[summary_source].loc[s].to_numpy()         
            padding = np.zeros((self.total_stations, ), dtype=np.float32)
            padding[i] = 1
            
            static = np.concatenate([static, padding], axis=0)
            
            self.train_static.extend([static for _ in range(x_train.shape[0])])
            self.test_static.extend([static for _ in range(x_test.shape[0])])
            
        self.train_timeseries = np.array(self.train_timeseries)
        self.test_timeseries = np.array(self.test_timeseries) 
        
        self.train_static = np.array(self.train_static)
        self.test_static = np.array(self.test_static)
        
        scaler = MinMaxScaler()
        scaler.fit(self.train_static)
        
        self.train_static = scaler.transform(self.train_static)
        self.test_static = scaler.transform(self.test_static)
                
        
        #self.train_static = np.ones((self.train_static.shape[0], self.train_static.shape[1]))
        
        self.train_y = np.array(self.train_y) 
        self.train_y = np.swapaxes(self.train_y, 1, 2)

        
        self.test_y = np.array(self.test_y)
        self.test_y = np.swapaxes(self.test_y, 1, 2)        

    def mapdata_tonumpy(self, map_ds):
        map_ds = map_ds.unbatch()

        x_array = []
        y_array = []

        for ts in map_ds:
            x = ts[0]
            y= ts[1]

            x_array.append(x.numpy())
            y_array.append(y.numpy())

        return np.array(x_array), np.array(y_array) 
    
    @property
    def train(self):
        return self.train_timeseries, self.train_static, self.train_y


    @property
    def test(self):      
        return self.test_timeseries, self.test_static, self.test_y


    def test_windows(self, station=None):
        total_size = self.test_array(station).shape[0]
        convolution_size = self.input_width
        prediction_size = self.label_width
        
        a = []
        
        for i in range(convolution_size, (total_size-prediction_size+1)):
            index_list = list(range(i, i+prediction_size))
            a.append(self.test_array(station)[index_list])
        
        return np.squeeze(array(a), axis=2)
    
    def test_array(self, station):
        return array(self.test_df[station][self.label_columns])


# In[11]:


class CustomLoss():  
    def qloss_95(y_true, y_pred, q=0.95):
        e = (y_true-y_pred)    
        return tf.square(y_true-y_pred) + K.maximum(q*e, (q-1)*e)
        
    def qloss_90(y_true, y_pred, q=0.9):
        e = (y_true-y_pred)    
        return tf.square(y_true-y_pred) + K.maximum(q*e, (q-1)*e)
    
    def qloss_70(y_true, y_pred, q=0.7):
        e = (y_true-y_pred)    
        return tf.square(y_true-y_pred) + K.maximum(q*e, (q-1)*e)
    
    def qloss_50(y_true, y_pred, q=0.5):
        e = (y_true-y_pred)    
        return tf.square(y_true-y_pred) + K.maximum(q*e, (q-1)*e)
    
class Model():
    MAX_EPOCHS = 150
    
    def __init__(self, window):
        # Store the raw data.
        self.window = window
        
        self.train_df = self.window.train_df
        self.test_df = self.window.test_df
             
    def compile_and_fit(self, model, window, loss_func, patience=10):

        model.compile(loss=loss_func,
                    optimizer=tf.optimizers.Adam(),
                    metrics=[tf.metrics.MeanAbsoluteError()])

        history = model.fit(window.train, epochs=self.MAX_EPOCHS,
                            verbose=0)

        return history

    def num_flood_events(self, cut=1):
        actuals = np.squeeze(self.window.test_windows, axis=2) 
        cut_percentile = np.percentile(actuals.flatten(), cut)
        locs = np.unique(np.where(actuals<cut_percentile)[0])

        events = np.split(locs, np.cumsum( np.where(locs[1:] - locs[:-1] > 1) )+1)

        return len(events)
    
    def summary(self, station=None):
        summary_dict = {}
        
        summary_dict['model_name'] = self.model_name
        summary_dict['input_width'] = self.window.input_width
        summary_dict['label_width'] = self.window.label_width
             
        if station == None:
            summary_dict['station'] = self.window.station
            summary_dict['inputs'] = str(list(self.train_df.columns))
            summary_dict['NSE'] = self.get_NSE()
        else:
            summary_dict['station'] = station
            example_station = self.train_df.columns.get_level_values(0)[0]
            summary_dict['inputs'] = str(list(self.train_df[example_station].columns))
            summary_dict['NSE'] = self.get_NSE(station)  
                             
        summary_dict['SER_1%'] = self.average_model_error(station, cut=1)
        summary_dict['SER_2%'] = self.average_model_error(station, cut=2)    
        summary_dict['SER_5%'] = self.average_model_error(station, cut=5)        
        summary_dict['SER_10%'] = self.average_model_error(station, cut=10)  
        summary_dict['SER_25%'] = self.average_model_error(station, cut=25)  
        summary_dict['SER_50%'] = self.average_model_error(station, cut=50)  
        summary_dict['SER_75%'] = self.average_model_error(station, cut=75)  
        summary_dict['RMSE'] = self.average_model_error(station, cut=100)
        

        summary_dict['f1_score_individual_1%'] = self.binary_metrics(station=station, cut=1, metric='f1_score', evaluation='individual')        
        summary_dict['f1_score_individual_2%'] = self.binary_metrics(station=station, cut=2, metric='f1_score', evaluation='individual')  
        summary_dict['f1_score_individual_5%'] = self.binary_metrics(station=station, cut=5, metric='f1_score', evaluation='individual') 
        summary_dict['f1_score_individual_10%'] = self.binary_metrics(station=station, cut=10, metric='f1_score', evaluation='individual') 
        summary_dict['f1_score_individual_25%'] = self.binary_metrics(station=station, cut=25, metric='f1_score', evaluation='individual') 
        summary_dict['f1_score_individual_50%'] = self.binary_metrics(station=station, cut=50, metric='f1_score', evaluation='individual')  
        summary_dict['f1_score_individual_75%'] = self.binary_metrics(station=station, cut=75, metric='f1_score', evaluation='individual') 
        summary_dict['f1_score_individual_all'] = self.binary_metrics(station=station, cut=100, metric='f1_score', evaluation='individual') 
          
        return summary_dict
            
    def print_model_error(self, station=None, cut=0):
        if station != None:
            preds = self.predictions(station)
            actuals = self.window.test_windows(station)
            test_array = self.window.test_array(station)
        else:
            preds = self.predictions(station)
            actuals = self.window.test_windows  
            test_array = self.window.test_array
            
        cut_percentile = np.percentile(actuals.flatten(), cut)

        locs = np.unique(np.where(actuals>cut_percentile)[0])
        preds = preds[locs]
        actuals = actuals[locs]

        for window_pred, window_actual, loc in zip(preds, actuals, locs):
            print("time: {}".format(loc))
            print("Input: {}".format(test_array[loc:loc+self.window.input_width].flatten()))
            print("Predicted: {}".format(window_pred))
            print("Actual: {}".format(window_actual))
            print("-------------------------")
            
    def model_predictions_less_than_cut(self, cut=100):
        
        preds = self.predictions
        actuals =self.window.test_windows

        cut_percentile = np.percentile(actuals.flatten(), cut)

        num_predicted = (preds.flatten() < cut_percentile).sum()
        num_actual = (actuals.flatten() < cut_percentile).sum()

        return num_predicted, num_actual
        
    def average_model_error(self, station=None, cut=100):
        if self.window.label_columns[0] == 'streamflow_MLd_inclInfilled':
            cut = 100 - cut
            
        if station != None:
            preds = self.predictions(station)
            actuals = self.window.test_windows(station)
        else:
            preds = self.predictions()
            actuals = self.window.test_windows         

        cut_percentile = np.percentile(actuals.flatten(), cut)

        locs = np.where(actuals>cut_percentile)[0]
        preds = preds[locs]
        actuals = actuals[locs]

        avg_error = 0

        for window_pred, window_actual in zip(preds, actuals):
            avg_error += np.sum((window_pred - window_actual)**2)
        

        avg_error = avg_error/actuals.shape[0]*actuals.shape[1]


        return avg_error
    
    def get_NSE(self, station=None, type='cast'):
        if station != None:
            preds = self.predictions(station)
            actuals = self.window.test_windows(station)
        else:
            preds = self.predictions()
            actuals = self.window.test_windows
        
        NSE = []

        for i in range(self.window.label_width):
            numer = np.sum(np.square(preds[:, i] - actuals[:, i]))
            denom = np.sum(np.square(actuals[:, i] - np.mean(actuals[:, i])))
        
            NSE.append(1-(numer/denom))
        
        if type=='cast':
            return np.mean(NSE)
        else:
            return NSE

    def binary_metrics(self, cut, metric, evaluation='whole', station=None):
        percentile_cut = self.window.station_percentile(station=station, cut=cut)
        
        if station==None:
            preds_pre = self.predictions()
            actuals_pre = self.window.test_windows
        else:        
            preds_pre = self.predictions(station)
            actuals_pre = self.window.test_windows(station)
            
        if evaluation=='whole':  
            preds = np.array([int(any(x > percentile_cut)) for x in preds_pre])
            actuals = np.array([int(any(x > percentile_cut)) for x in actuals_pre])
        else:
            preds = np.array([int(x > percentile_cut) for x in preds_pre.flatten()])           
            actuals = np.array([int(x > percentile_cut) for x in actuals_pre.flatten()])

        if metric=='accuracy':
            return accuracy_score(actuals, preds)
        elif metric=='precision':
            return precision_score(actuals, preds)
        elif metric=='recall':
            return recall_score(actuals, preds)
        elif metric=='f1_score':
            return f1_score(actuals, preds)
     

    @property
    def test_loss(self):
        return self.model.evaluate(self.window.test, verbose=0)[0]

    def predictions(self, station=None):
        tf_test = self.window.test

        if station != None:
            filter_index = self.window.stations.index(station)
            num_inputs = len(self.window.train_df.columns.levels[1])
            tf_test = tf_test.unbatch().filter(lambda x, y: tf.math.reduce_sum(x[:, num_inputs + filter_index]) > 0).batch(32)

        return np.squeeze(self.model.predict(tf_test), axis=2)
        
class Base_Model(Model):  
    def __init__(self, model_name, window, CONV_WIDTH, output_activation='sigmoid', loss_func=tf.losses.MeanSquaredError()):
        super().__init__(window)
        
        self.model_name = model_name
        self.mix_type_name = None
        self.loss_func = loss_func
        
        if self.model_name == 'multi-linear':          
            self.model = tf.keras.Sequential([
                            # Take the last time step.
                            # Shape [batch, time, features] => [batch, 1, features]
                            tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
                            # Shape => [batch, 1, dense_units]
                            tf.keras.layers.Dense(20, activation='relu'),
                            # Shape => [batch, out_steps*features]
                            tf.keras.layers.Dense(CONV_WIDTH, activation=output_activation, 
                                                  kernel_initializer=tf.initializers.zeros()),
                            # Shape => [batch, out_steps, features=1]
                            tf.keras.layers.Reshape([CONV_WIDTH, 1])
                        ])
            
        elif self.model_name == 'multi-CNN':
            self.model = tf.keras.Sequential([
                            # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
                            tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
                            # Shape => [batch, 1, conv_units]
                            tf.keras.layers.Conv1D(64, activation='relu', kernel_size=(CONV_WIDTH)),
                            # Shape => [batch, 1,  out_steps*features]
                            tf.keras.layers.Dense(CONV_WIDTH, activation=output_activation, 
                                                  kernel_initializer=tf.initializers.zeros()),
                            # Shape => [batch, out_steps, features=1]
                            tf.keras.layers.Reshape([CONV_WIDTH, 1])
                        ])
            
        elif self.model_name == 'multi-LSTM':                       
            self.model = Sequential([
                            # Shape [batch, time, features] => [batch, lstm_units].
                            # Adding more `lstm_units` just overfits more quickly.
                            LSTM(20, return_sequences=False),
                            # Shape => [batch, out_steps*features].
                            Dense(CONV_WIDTH, activation=output_activation,
                                                  kernel_initializer=tf.initializers.zeros()),
                            # Shape => [batch, out_steps, features=1].
                            Reshape([CONV_WIDTH, 1])
                        ])

            
        elif self.model_name == 'multi-ED-LSTM':                       
            self.model = Sequential([
                            # Shape [batch, time, features] => [batch, lstm_units].
                            # Adding more `lstm_units` just overfits more quickly.
                            LSTM(20, return_sequences=True,),
                            # Shape => [batch, out_steps*features].
                            Dropout(0.2),
                            Flatten(),
                            RepeatVector(5),
                            LSTM(20, return_sequences=False), 
                            
                            Dropout(0.2),                
                            Dense(CONV_WIDTH, activation=output_activation,
                                                  kernel_initializer=tf.initializers.zeros()),
                            # Shape => [batch, out_steps, features=1].
                            Reshape([CONV_WIDTH, 1])
                        ])    
            
        elif self.model_name == 'multi-Bidirectional-LSTM':                       
            self.model = Sequential([
                            # Shape [batch, time, features] => [batch, lstm_units].
                            # Adding more `lstm_units` just overfits more quickly.
                            Bidirectional(LSTM(20, return_sequences=False)),
                            # Shape => [batch, out_steps*features].
                            Dense(CONV_WIDTH, activation=output_activation,
                                                  kernel_initializer=tf.initializers.zeros()),
                            # Shape => [batch, out_steps, features=1].
                            Reshape([CONV_WIDTH, 1])
                        ])  
            
        elif self.model_name == 'multi-deep-Bidirectional-LSTM':                       
            self.model = Sequential([
                            # Shape [batch, time, features] => [batch, lstm_units].
                            # Adding more `lstm_units` just overfits more quickly.
                            Bidirectional(LSTM(64, return_sequences=True
                                              )),
                            Dropout(0.2),
                            Bidirectional(LSTM(32, return_sequences=False)),
                            Dropout(0.2),
                            # Shape => [batch, out_steps*features].
                            Dense(CONV_WIDTH, activation=output_activation,
                                                  kernel_initializer=tf.initializers.zeros()),
                            # Shape => [batch, out_steps, features=1].
                            Reshape([CONV_WIDTH, 1])
                        ]) 
            
        self.compile_and_fit(self.model, window, loss_func)


# In[12]:


class Mixed_Model(Base_Model):
    threshold = 0.2
    
    def __init__(self, model_name, mix_type_name, window, CONV_WIDTH):
        super().__init__(model_name, window, CONV_WIDTH)
        self.mix_type_name = mix_type_name
               
        if self.mix_type_name == 'simple-two_model-onestepAR':
            window_simple = WindowGenerator(input_width=1,
                                             label_width=1,
                                             shift=1,
                                             train_df=train_df.loc[:,train_df.columns.get_level_values(1).isin(self.window.label_columns)] ,
                                             test_df=test_df.loc[:,test_df.columns.get_level_values(1).isin(self.window.label_columns)] ,
                                             station=self.window.station,
                                             label_columns=['flood_probabilities'])
            
            self.model_simple = Base_Model(model_name=model_name, window=window_simple, CONV_WIDTH=1)
            
        elif self.mix_type_name == 'simple-two_model-multistep':
            window_simple = WindowGenerator(input_width=1,
                                             label_width=self.window.label_width,
                                             shift=self.window.label_width,
                                             train_df=train_df,
                                             test_df=test_df,
                                             station=self.window.station,
                                             label_columns=['flood_probabilities'])
            
            self.model_simple = Base_Model(model_name=model_name, window=window_simple, CONV_WIDTH=self.window.label_width)
        elif self.mix_type_name == 'upper_soil-two_model-multistep':
            window_simple = WindowGenerator(input_width=self.window.input_width,
                                             label_width=self.window.label_width,
                                             shift=self.window.label_width,
                                             train_df=train_df,
                                             test_df=test_df,
                                             station=self.window.station,
                                            filtered='upper_soil_filter',
                                             label_columns=['flood_probabilities'])
            
            self.model_simple = Base_Model(model_name=model_name, window=window_simple, CONV_WIDTH=self.window.label_width)
            
            
    @property
    def predictions(self):
        if self.mix_type_name == 'simple':
            preds = super().predictions
            test_array = self.window.test_array[self.window.input_width:]
            new_pred=[]
       
            for pred, actual_before in zip(preds, test_array):
                if actual_before < self.threshold:
                    pred = np.full((self.window.label_width,), actual_before)

                new_pred.append(pred)  
            
            
            
            return np.array(new_pred)
        
        elif self.mix_type_name == 'simple-two_model-onestepAR':
            preds = super().predictions
            preds_simple = self.model_simple.predictions
            
            # test array starts 1 time unit before predictions
            test_array = self.window.test_array[self.window.input_width:]
            
            new_pred=[]

            for pred, actual_before in zip(preds, test_array):
                if actual_before < self.threshold:
                    pred = []
                                      
                    input_value = np.array(actual_before).reshape(1,1,1)
                    
                    for j in range(self.window.label_width):
                        pred_simple = self.model_simple.model.predict(input_value).item()
                        pred.append(pred_simple)
                        
                        input_value = np.array(pred_simple).reshape(1,1,1)
                                         
                    pred = np.array(pred)

                new_pred.append(pred)  
            
            return np.array(new_pred)
        
        elif self.mix_type_name == 'simple-two_model-multistep':
            preds = super().predictions
            preds_simple = self.model_simple.predictions
            
            # test array starts 1 time unit before predictions
            test_array = self.window.test_array[self.window.input_width:]
            
            new_pred=[]

            for i, (pred, actual_before) in enumerate(zip(preds, test_array)):
                if actual_before < self.threshold:                                
                    input_value = self.window.test_example(i+self.window.input_width)
                                              
                    pred = self.model_simple.model.predict(input_value).flatten()

                new_pred.append(pred)  
            
            return np.array(new_pred)
        
        elif self.mix_type_name == 'upper_soil-two_model-multistep':
            preds = super().predictions
            preds_simple = self.model_simple.predictions
            
            # upper soil indicator 1 time unit before predictions
            upper_soil_indicator = window.test_indicator(filtered='upper_soil_filter')
                      
            new_pred=[]

            for i, (pred, indicator) in enumerate(zip(preds, upper_soil_indicator)):
                if indicator == 1:                                
                    input_value = self.window.test_example(i+self.window.input_width)
                                              
                    pred = self.model_simple.model.predict(input_value).flatten()

                new_pred.append(pred)              
            
            return np.array(new_pred)  


# In[13]:


## Without Changes

class Ensemble_Static():
    epochs = 100
    patience = 5
    def __init__(self, numpy_window, batch_size=32):
        num_timesteps = numpy_window.input_width
        num_timeseries_features = numpy_window.num_timeseries_features
        num_static_features = numpy_window.num_static_features + numpy_window.total_stations
          
        num_predictions = numpy_window.label_width
        
        self.batch_size = batch_size
        self.stations = numpy_window.stations
        self.n_stations = numpy_window.total_stations
        self.numpy_window = numpy_window
        # RNN + SLP Model
        # Define input layer

        recurrent_input = Input(shape=(num_timesteps, num_timeseries_features),name="TIMESERIES_INPUT")
        static_input = Input(shape=(num_static_features,),name="STATIC_INPUT")

        # RNN Layers
        # layer - 1
        rec_layer_one = LSTM(20, name ="BIDIRECTIONAL_LAYER_1", return_sequences=True)(recurrent_input)
        rec_layer_one = Dropout(0.1,name ="DROPOUT_LAYER_1")(rec_layer_one)
        
        # layer - 2
        rec_layer_two = LSTM(20, name ="BIDIRECTIONAL_LAYER_2", return_sequences=False)(rec_layer_one)
        rec_layer_two = Dropout(0.1,name ="DROPOUT_LAYER_2")(rec_layer_two)      
        

        # SLP Layers
        static_layer_one = Dense(20, activation='relu',name="DENSE_LAYER_1")(static_input)
        # Combine layers - RNN + SLP
        combined = Concatenate(axis= 1,name = "CONCATENATED_TIMESERIES_STATIC")([rec_layer_two, static_layer_one])
        combined_dense_two = Dense(20, activation='relu',name="DENSE_LAYER_2")(combined)
        output = Dense(num_predictions, name="OUTPUT_LAYER", activation='sigmoid')(combined_dense_two)

      
        # Compile ModeL
        self.model = keras.models.Model(inputs=[recurrent_input, static_input], outputs=[output])
        # MSE
        
        #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.train_timeseries_x, self.train_static_x, self.train_y = numpy_window.train     
        self.test_timeseries_x, self.test_static_x, self.test_y = numpy_window.test 
        
        self.model.summary()
        
    def train(self):
        self.model.compile(loss='MeanSquaredError', optimizer='adam', metrics=['MeanAbsoluteError'])
        
        
        self.model.fit([self.train_timeseries_x, self.train_static_x], 
                       self.train_y, 
                       epochs=self.epochs, 
                       batch_size=self.batch_size, 
                       verbose=1)


    @property
    def test_loss(self):
        return self.model.evaluate(self.window.test, verbose=0)[0]
    
    def predictions(self, station):  
        filter_index = self.stations.index(station)
        n_observations = int(self.test_static_x.shape[0]/self.n_stations)

        start = int(filter_index*n_observations)
        end = int((filter_index+1)*n_observations)
        
        print("Timeseries input shape:", self.test_timeseries_x[start:end].shape)
        print("Static input shape:", self.test_static_x[start:end].shape)
        
        return self.model.predict([self.test_timeseries_x[start:end,:], self.test_static_x[start:end,:]])
    
    
    def actuals(self, station):
        filter_index = self.stations.index(station)
        n_observations = int(self.test_static_x.shape[0]/self.n_stations)

        start = int(filter_index*n_observations)
        end = int((filter_index+1)*n_observations)
        print('TSET Y SHAPE',self.test_y.shape)
        

        return self.test_y.reshape(self.test_y.shape[0], self.test_y.shape[2])[start:end, :]

    
    def average_model_error(self, station, cut=100):
        preds = self.predictions(station)
        actuals = self.actuals(station)
        
        cut_percentile = np.percentile(actuals.flatten(), 100-cut)

        locs = np.where(actuals > cut_percentile)[0]
        preds = preds[locs]
        actuals = actuals[locs]

        avg_error = 0

        for window_pred, window_actual in zip(preds, actuals):
            avg_error += np.sum((window_pred - window_actual)**2)
        
        if avg_error==0:
            return 0
        print(avg_error)
        avg_error = avg_error/actuals.shape[0]*actuals.shape[1]

        return avg_error 
    
    def print_model_windows(self, station, cut=100):
        preds = self.predictions(station)
        actuals = self.actuals(station)
        cut_percentile = np.percentile(actuals.flatten(), 100-cut)

        locs = np.where(actuals > cut_percentile)[0]
        preds = preds[locs]
        actuals = actuals[locs]
        
        for pred, actual, loc in zip(preds, actuals, locs):
            print("time: {}".format(loc))
            print("Input: {}".format(self.test_y[loc:loc+self.numpy_window.input_width+1].flatten()))
            print("Predicted: {}".format(pred))
            print("Actual: {}".format(actual))
            print("-------------------------")        

    def summary(self, station):
        summary_dict = {}
        
        summary_dict['station'] = station
        summary_dict['input_width'] = self.numpy_window.input_width
        summary_dict['label_width'] = self.numpy_window.label_width
        summary_dict['num_timeseries_features'] = self.numpy_window.num_timeseries_features 
        summary_dict['num_static_features'] = self.numpy_window.num_static_features        
        summary_dict['timeseries_inputs'] = self.numpy_window.timeseries_source
        summary_dict['static_inputs'] = self.numpy_window.summary_source     

   
        summary_dict['SERA_1%'] = self.average_model_error(station, cut=1)
        summary_dict['SERA_2%'] = self.average_model_error(station, cut=2)    
        summary_dict['SERA_5%'] = self.average_model_error(station, cut=5)        
        summary_dict['SERA_10%'] = self.average_model_error(station, cut=10)  
        summary_dict['SERA_25%'] = self.average_model_error(station, cut=25)  
        summary_dict['SERA_50%'] = self.average_model_error(station, cut=50)  
        summary_dict['SERA_75%'] = self.average_model_error(station, cut=75)  
        summary_dict['SERA_all'] = self.average_model_error(station, cut=100)
        
        
        
        
        return summary_dict


# In[14]:


class Switch_Model(Model):
    threshold = 0.7
    
    def __init__(self, window_switch, window_regular, CONV_WIDTH):
        self.window_switch = window_switch
        self.window = window_regular
        
        assert(window_switch.input_width == self.window.input_width)
        
        self.switch = Ensemble_Static(window_switch)
        
        self.regular = Base_Model(model_name='multi-LSTM', window=window_regular, CONV_WIDTH=CONV_WIDTH)
        self.q70 = Base_Model(model_name='multi-LSTM', window=window_regular, CONV_WIDTH=CONV_WIDTH, loss_func=CustomLoss.qloss_70)
        self.q95 = Base_Model(model_name='multi-LSTM', window=window_regular, CONV_WIDTH=CONV_WIDTH, loss_func=CustomLoss.qloss_95)
        
    def predictions(self, station):
        preds_switch = self.switch.predictions(station)     
        
        preds_regular = self.regular.predictions(station)
        preds_q70 = self.q70.predictions(station)        
        preds_q95 = self.q95.predictions(station)
        
        test_array = self.window.test_windows(station)   

        new_pred=[]
        
        for pred_switch, pred_regular, pred_q70, pred_q95 in zip(preds_switch, preds_regular, preds_q70, preds_q95):

                
            switch_condition = pred_switch > 0.95
            q95_condition = pred_switch > 0.7
            q70_condition = pred_switch <= 0.7  # You might want to specify this condition differently

            new_pred.append(np.where(switch_condition, pred_q95, np.where(q95_condition, pred_q70, pred_regular)))
                
        return np.array(new_pred)
        

    def test_MSE(self, station=None):
        preds = self.predictions(data='test', station=station)
        test_array = self.window.test_array(station)[self.window.input_width:]

        return mean_squared_error(test_array, preds)
    
    def test_ROCAUC(self, station, level=0.05):
        preds = self.predictions(data='test', station=station)
        test_array = (self.window.test_array(station)[self.window.input_width:] < level).astype(int)
        
        return roc_auc_score(test_array, preds)

    def summary(self, station=None):
        summary_dict = {}
        
        summary_dict['input_width'] = self.window.input_width
        summary_dict['label_width'] = self.window.label_width
        
        summary_dict['station'] = station

        summary_dict['NSE'] = self.get_NSE(station)       
                  
        summary_dict['SER_1%'] = self.average_model_error(station, cut=1)
        summary_dict['SER_2%'] = self.average_model_error(station, cut=2)    
        summary_dict['SER_5%'] = self.average_model_error(station, cut=5)        
        summary_dict['SER_10%'] = self.average_model_error(station, cut=10)  
        summary_dict['SER_25%'] = self.average_model_error(station, cut=25)  
        summary_dict['SER_50%'] = self.average_model_error(station, cut=50)  
        summary_dict['SER_75%'] = self.average_model_error(station, cut=75)  
        summary_dict['RMSE'] = self.average_model_error(station, cut=100)
        
        return summary_dict   


# In[15]:


camels_data.summary_data= camels_data.summary_data.T.drop_duplicates().T


# In[16]:


camels_data.summary_data['state_outlet'].value_counts()


# ### Plotting catchments on Map

# In[17]:


cd_sd = camels_data.summary_data.loc[:,~camels_data.summary_data.columns.duplicated()]


# In[18]:


import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the list of cities and their latitudes/longitudes
cities = cd_sd['station_name']
lats = cd_sd['lat_outlet']
longs = cd_sd['long_outlet']

# Generate a random priority for each city between 1 and 5
priority = np.random.randint(1, 6, size=len(cities))
state= cd_sd['state_outlet']

# Create the DataFrame with the city data
data = {'cityname': cities,
        'lats': lats,
        'longs': longs,
        'States': state,
        'priority': priority
        }

df = pd.DataFrame(data)


# In[19]:


state_mapping = {'QLD': 1, 'NSW': 2, 'SA': 3, 'VIC': 4, 'ACT': 5, 'WA': 6, 'NT': 7, 'TAS': 8}
df['state_num'] = df['States'].map(state_mapping)


# In[39]:


# Load the shapefile of Australia
australia = gpd.read_file('STE_2021_AUST_SHP_GDA2020/STE_2021_AUST_GDA2020.shp')

# Define the CRS of the shapefile manually
australia.crs = 'epsg:7844'

# Create a GeoDataFrame from the DataFrame of cities
gdf_cities = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longs, df.lats))

# Set the CRS of the GeoDataFrame to EPSG 7844
# https://epsg.io/7844
gdf_cities.crs = 'epsg:7844'

# Reproject the GeoDataFrame of cities to match the CRS of the shapefile
gdf_cities = gdf_cities.to_crs(australia.crs)

# Perform a spatial join to link the cities to their corresponding polygons in the shapefile
gdf_cities = gpd.sjoin(gdf_cities, australia, predicate='within')

# Set up the plot
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(8, 8))

# # Define a custom dark color palette
custom_palette = sns.color_palette(['darkblue', 'black', 'purple','darkred', 'darkgreen', 'darkorange', 'brown' , 'blue'], n_colors=len(df['state_num'].unique()))

# Plot the cities colored by priority with adjustments
sns.scatterplot(ax=ax, data=gdf_cities, x='longs', y='lats', hue='States', s=15, palette=custom_palette, edgecolor='black', alpha=0.8, legend='full', zorder=2)


# Set x-axis limits
ax.set_xlim(110, 160)

# Add the shapefile of Australia as a background map
australia.plot(ax=ax, color='lightgrey', edgecolor='white', zorder=1)

# Set the plot title and axis labels
plt.title('Catchments across Australia')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Show the plot
plt.show()


# ### Conditional Ensemble

# In[17]:


camels_data.summary_data['state_outlet'].value_counts()


# In[18]:


selected_stations = list(camels_data.summary_data[camels_data.summary_data['state_outlet'] == 'WA'].index)


# In[19]:


#, 'upper_soil', 'deep_soil'
combined=[]
for i in range(0,1):
    print('RUN',i)
    results_switch=[]
    variable_ts = ['streamflow_MLd_inclInfilled', 'precipitation_deficit', 'year_sin', 'year_cos', 'tmax_AWAP', 'tmin_AWAP']
    variable_ts_switch = ['flood_probabilities', 'precipitation_deficit', 'year_sin', 'year_cos', 'tmax_AWAP', 'tmin_AWAP']

    variable_static = ['q_mean', 'stream_elas', 'runoff_ratio', 'high_q_freq', 'high_q_dur', 'low_q_freq', 'zero_q_freq']

    train_df, test_df = camels_data.get_train_val_test(source=variable_ts, stations=selected_stations)

    multi_window = MultiWindow(input_width=5,
                               label_width=5,
                               shift=5,
                               train_df=train_df,
                               test_df=test_df,
                               stations=selected_stations,
                               label_columns=['streamflow_MLd_inclInfilled'])

    np_window = MultiNumpyWindow(input_width=5, 
                                 label_width=5,
                                 shift=5,
                                 timeseries_source=variable_ts_switch,
                                 summary_source=variable_static,
                                 summary_data=camels_data.summary_data,
                                 stations=selected_stations,
                                 label_columns=['flood_probabilities'])

    model_switch = Switch_Model(window_switch=np_window, window_regular=multi_window, CONV_WIDTH=5) 

    for station in selected_stations:
                results_switch.append(model_switch.summary(station))
    
    Switch_NSW= pd.DataFrame(results_switch)
    Switch_NSW= Switch_NSW.mean()
    Switch_NSW= Switch_NSW.to_dict()
    combined.append(Switch_NSW)


# In[23]:


Switch3_QLD_bdlstm = pd.DataFrame(combined)
Switch3_QLD_bdlstm= Switch3_QLD_bdlstm.to_csv('WOV_Switch1_WA_lstm_3.csv')


# In[21]:


df_date= np_window.test_df['604053'].reset_index()
date_values= df_date['date'][5:-4]


# In[28]:


# for dates

import matplotlib.patches as mpatches
fig = plt.figure(figsize=(20,5))
plt.title('WA-604053', fontsize= 20)
# plt.xlabel('')
plt.ylabel('Flood Probability', fontsize=18)
plt.ylim(0, 1.1)

plt.rcParams.update({'font.size': 15})
# plt.set_xlabel(fontsize=15)
# plt.set_ylabel('Flood Probability', fontsize=15)

s=300
e=1100

ax1 = plt.plot(date_values[s:e], model_switch.predictions('604053')[:,0][s:e], color='blue')
ax2 = plt.plot(date_values[s:e], multi_window.test_windows('604053')[:,0][s:e], color='red')

red_patch = mpatches.Patch(color='red', label='Actual')
blue_patch = mpatches.Patch(color='blue', label='Predicted')

plt.legend(handles=[red_patch, blue_patch])
plt.show()


# ### Individual - SA - All Models

# In[21]:


selected_stations = list(camels_data.summary_data[camels_data.summary_data['state_outlet'] == 'NSW'].index)


# In[22]:


#['multi-LSTM', 'multi-linear','multi-CNN', 'multi-Bidirectional-LSTM']

combined= []
for i in range(0,10):
    print('RUN', i)
    input_widths = [5]
    label_widths = [5]
    models = ['multi-LSTM']
    variables = [['streamflow_MLd_inclInfilled', 'precipitation_deficit', 'year_sin', 'year_cos', 'tmax_AWAP', 'tmin_AWAP']]

    permutations_base = list(itertools.product(*[input_widths, label_widths, selected_stations, models, variables]))

    results_baseModels_variables = []
    models_baseModels_variables = []
    errors_baseModels_variables = []

    for input_width, label_width, station, model_name, variable in permutations_base:
        if input_width < label_width:
            continue

        train_df, test_df = camels_data.get_train_val_test(source=variable, stations=selected_stations)

        try:
            print('input_width:{}, label_width:{}, station:{}, model:{}, variables:{}'.format(input_width, label_width, station, model_name, variable))
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(current_time)

            window = WindowGenerator(input_width=input_width,
                                     label_width=label_width,
                                     shift=label_width,
                                     train_df=train_df,
                                     test_df=test_df,
                                     station=station,
                                     label_columns=['streamflow_MLd_inclInfilled'])

            model = Base_Model(model_name=model_name, window=window, CONV_WIDTH=label_width)

            results_baseModels_variables.append(model.summary())

            pd.DataFrame(results_baseModels_variables).to_csv('results_files/results_ensemble_all_1.csv')

        except:
            errors_baseModels_variables.append([input_width, label_width, station, model])

    Individual_SA= pd.DataFrame(results_baseModels_variables)
    Individual_SA= Individual_SA.mean()
    Individual_SA= Individual_SA.to_dict()
    combined.append(Individual_SA)


# In[23]:


Individual_SA_bdlstm = pd.DataFrame(combined)
Individual_SA_bdlstm= Individual_SA_bdlstm.to_csv('WOV_Individual_NSW_lstm_1.csv')


# ### Batch-Indicator LSTM- SA Only

# In[20]:


selected_stations = list(camels_data.summary_data[camels_data.summary_data['state_outlet'] == 'SA'].index)


# In[21]:


len(selected_stations)


# In[24]:


combined= []
for i in range(0,30):
    print('RUN',i)
    input_widths = [5]
    label_widths = [5]
    models = ['multi-LSTM']
    variables = [['streamflow_MLd_inclInfilled', 'precipitation_deficit', 'year_sin', 'year_cos', 'tmax_AWAP', 'tmin_AWAP']]

    permutations_base_batch = list(itertools.product(*[input_widths, label_widths, models, variables]))

    results_baseModels_batch = []
    errors_baseModels_batch = []

    for input_width, label_width, model_name, variable in permutations_base_batch:
        try:
            if input_width < label_width:
                continue

            train_df, test_df = camels_data.get_train_val_test(source=variable, stations=selected_stations, discard=0.5)
            multi_window = MultiWindow(input_width=input_width,
                                       label_width=label_width,
                                       shift=label_width,
                                       train_df=train_df,
                                       test_df=test_df,
                                       stations=selected_stations,
                                       label_columns=['streamflow_MLd_inclInfilled'])

            model = Base_Model(model_name=model_name, window=multi_window, CONV_WIDTH=label_width)

            print('input_width:{}, label_width:{}, model:{}, variables:{}'.format(input_width, label_width, model_name, variable))
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(current_time)

            for station in selected_stations:
                results_baseModels_batch.append(model.summary(station=station))

            pd.DataFrame(results_baseModels_batch).to_csv('results_files/results_batch_all_1.csv')
        except:
            errors_baseModels_batch.append([input_width, label_width, model_name]) 
    Batch_NSW= pd.DataFrame(results_baseModels_batch)
    Batch_NSW= Batch_NSW.mean()
    Batch_NSW= Batch_NSW.to_dict()
    combined.append(Batch_NSW)
    


# In[25]:


Batch_NSW_LSTM = pd.DataFrame(combined)
Batch_NSW_LSTM.to_csv('WOV_Batch_SA_LSTM.csv')


# ### Batch-Static SA- LSTM

# In[26]:


selected_stations = list(camels_data.summary_data[camels_data.summary_data['state_outlet'] == 'SA'].index)


# In[27]:


len(selected_stations)


# In[29]:


combined=[]
for i in range(0,30):
    print('RUN', i)
    input_widths = [5]
    label_widths = [5]
    models = ['multi-LSTM']
    variables = [['streamflow_MLd_inclInfilled', 'precipitation_deficit', 'year_sin', 'year_cos', 'tmax_AWAP', 'tmin_AWAP', 'q_mean', 'stream_elas', 'runoff_ratio', 'high_q_freq', 'high_q_dur', 'low_q_freq', 'zero_q_freq']]

    permutations_base_batch = list(itertools.product(*[input_widths, label_widths, models, variables]))

    results_baseModels_batch = []
    errors_baseModels_batch = []

    for input_width, label_width, model_name, variable in permutations_base_batch:
        try:
            if input_width < label_width:
                continue

            train_df, test_df = camels_data.get_train_val_test(source=variable, stations=selected_stations)
            multi_window = MultiWindow(input_width=input_width,
                                       label_width=label_width,
                                       shift=label_width,
                                       train_df=train_df,
                                       test_df=test_df,
                                       stations=selected_stations,
                                       label_columns=['streamflow_MLd_inclInfilled'])

            model = Base_Model(model_name=model_name, window=multi_window, CONV_WIDTH=label_width)

            print('input_width:{}, label_width:{}, model:{}, variables:{}'.format(input_width, label_width, model_name, variable))
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(current_time)

            for station in selected_stations:
                results_baseModels_batch.append(model.summary(station=station))

            pd.DataFrame(results_baseModels_batch).to_csv('results_files/results_batch_static_all_1.csv')
        except:
            errors_baseModels_batch.append([input_width, label_width, model])
    Batch_NSW= pd.DataFrame(results_baseModels_batch)
    Batch_NSW= Batch_NSW.mean()
    Batch_NSW= Batch_NSW.to_dict()
    combined.append(Batch_NSW)


# In[30]:


Batch_NSW_LSTM = pd.DataFrame(combined)
Batch_NSW_LSTM.to_csv('WOV_BatchStatic_SA_LSTM.csv')


# ### Ensemble Model

# In[23]:


selected_stations = list(camels_data.summary_data[camels_data.summary_data['state_outlet'] == 'SA'].index)


# In[24]:


len(selected_stations)


# In[27]:


combined=[]
for i in range(0,15):
    print('RUN:', i)
    results_ensemble = []
    input_widths = [5]
    label_widths = [5]

    permutations_ensemble = list(itertools.product(*[input_widths, label_widths]))

    for input_width, label_width in permutations_ensemble:
        np_window = MultiNumpyWindow(input_width=input_width, 
                                     label_width=label_width,
                                     shift=label_width,
                                     timeseries_source=['streamflow_MLd_inclInfilled', 'precipitation_deficit', 'year_sin', 'year_cos', 'tmax_AWAP', 'tmin_AWAP'],
                                     summary_source=['q_mean', 'stream_elas', 'runoff_ratio', 'high_q_freq', 'high_q_dur', 'low_q_freq', 'zero_q_freq'],
                                     summary_data=camels_data.summary_data,
                                     stations=selected_stations,
                                     label_columns=['streamflow_MLd_inclInfilled'])

        ensemble_model = Ensemble_Static(np_window)
        ensemble_model.train()
        print('done')

        for station in selected_stations:
            results_ensemble.append(ensemble_model.summary(station))
                
    Ensemble_NSW= pd.DataFrame(results_ensemble)
    Ensemble_NSW= Ensemble_NSW.mean()
    Ensemble_NSW= Ensemble_NSW.to_dict()
    combined.append(Ensemble_NSW)


# In[28]:


Ensemble_NSW_LSTM = pd.DataFrame(combined)
Ensemble_NSW_LSTM.to_csv('WOV_Ensemble_SA_LSTM_2.csv')


# ### Batch Model

# In[ ]:


selected_stations = list(camels_data.summary_data[camels_data.summary_data['state_outlet'] == 'VIC'].index)


# In[ ]:


selected_stations


# In[ ]:


len(selected_stations)


# In[ ]:


input_widths = [5]
label_widths = [5]
models = ['multi-Bidirectional-LSTM']
variables = [['streamflow_MLd_inclInfilled', 'precipitation_deficit', 'year_sin', 'year_cos', 'tmax_AWAP', 'tmin_AWAP']]

permutations_base_batch = list(itertools.product(*[input_widths, label_widths, models, variables]))

results_baseModels_batch = []
errors_baseModels_batch = []

for input_width, label_width, model_name, variable in permutations_base_batch:
    try:
        if input_width < label_width:
            continue

        train_df, val_df, test_df = camels_data.get_train_val_test(source=variable, stations=selected_stations, discard=0.5)
        multi_window = MultiWindow(input_width=input_width,
                                   label_width=label_width,
                                   shift=label_width,
                                   train_df=train_df,
                                   val_df=val_df,
                                   test_df=test_df,
                                   stations=selected_stations,
                                   label_columns=['streamflow_MLd_inclInfilled'])

        model = Base_Model(model_name=model_name, window=multi_window, CONV_WIDTH=label_width)

        print('input_width:{}, label_width:{}, model:{}, variables:{}'.format(input_width, label_width, model_name, variable))
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(current_time)

        for station in selected_stations:
            results_baseModels_batch.append(model.summary(station=station))
         
        pd.DataFrame(results_baseModels_batch).to_csv('results_files/results_batch_all_1.csv')
    except:
        errors_baseModels_batch.append([input_width, label_width, model_name]) 


# In[ ]:


len(results_baseModels_batch)


# In[ ]:


Batch_VIC_bilstm= pd.DataFrame(results_baseModels_batch)
Batch_VIC_bilstm.to_csv('Batch_VIC_bilstm.csv')


# In[ ]:





# ### Individual

# In[ ]:


selected_stations = list(camels_data.summary_data[camels_data.summary_data['state_outlet'] == 'SA'].index)


# In[ ]:


selected_stations


# In[ ]:


len(selected_stations)


# In[ ]:


input_widths = [5]
label_widths = [5]
models = ['multi-LSTM', 'multi-linear','multi-CNN', 'multi-Bidirectional-LSTM']
variables = [['streamflow_MLd_inclInfilled', 'precipitation_deficit', 'year_sin', 'year_cos', 'tmax_AWAP', 'tmin_AWAP']]

permutations_base = list(itertools.product(*[input_widths, label_widths, selected_stations, models, variables]))
 
results_baseModels_variables = []
models_baseModels_variables = []
errors_baseModels_variables = []

for input_width, label_width, station, model_name, variable in permutations_base:
    if input_width < label_width:
        continue
        
    train_df, val_df, test_df = camels_data.get_train_val_test(source=variable, stations=selected_stations)
     
    try:
        print('input_width:{}, label_width:{}, station:{}, model:{}, variables:{}'.format(input_width, label_width, station, model_name, variable))
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(current_time)
        
        window = WindowGenerator(input_width=input_width,
                                 label_width=label_width,
                                 shift=label_width,
                                 train_df=train_df,
                                 val_df=val_df,
                                 test_df=test_df,
                                 station=station,
                                 label_columns=['streamflow_MLd_inclInfilled'])

        model = Base_Model(model_name=model_name, window=window, CONV_WIDTH=label_width)
        
        results_baseModels_variables.append(model.summary())
          
        pd.DataFrame(results_baseModels_variables).to_csv('results_files/results_ensemble_all_1.csv')
        
    except:
        errors_baseModels_variables.append([input_width, label_width, station, model])


# In[ ]:


len(results_baseModels_variables)


# In[ ]:


Individual_SA= pd.DataFrame(results_baseModels_variables)
Individual_SA.to_csv('Individual_SA.csv')


# ### Batch Static

# In[ ]:


selected_stations = list(camels_data.summary_data[camels_data.summary_data['state_outlet'] == 'ACT'].index)


# In[ ]:


len(selected_stations)


# In[ ]:


selected_stations


# In[ ]:


input_widths = [5]
label_widths = [5]
models = ['multi-LSTM', 'multi-linear','multi-CNN', 'multi-Bidirectional-LSTM']
variables = [['streamflow_MLd_inclInfilled', 'precipitation_deficit', 'year_sin', 'year_cos', 'tmax_AWAP', 'tmin_AWAP', 'q_mean', 'stream_elas', 'runoff_ratio', 'high_q_freq', 'high_q_dur', 'low_q_freq', 'zero_q_freq']]

permutations_base_batch = list(itertools.product(*[input_widths, label_widths, models, variables]))

results_baseModels_batch = []
errors_baseModels_batch = []

for input_width, label_width, model_name, variable in permutations_base_batch:
    try:
        if input_width < label_width:
            continue

        train_df, val_df, test_df = camels_data.get_train_val_test(source=variable, stations=selected_stations, discard=0.5)
        multi_window = MultiWindow(input_width=input_width,
                                   label_width=label_width,
                                   shift=label_width,
                                   train_df=train_df,
                                   val_df=val_df,
                                   test_df=test_df,
                                   stations=selected_stations,
                                   label_columns=['streamflow_MLd_inclInfilled'])

        model = Base_Model(model_name=model_name, window=multi_window, CONV_WIDTH=label_width)

        print('input_width:{}, label_width:{}, model:{}, variables:{}'.format(input_width, label_width, model_name, variable))
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(current_time)

        for station in selected_stations:
            results_baseModels_batch.append(model.summary(station=station))
         
        pd.DataFrame(results_baseModels_batch).to_csv('results_files/results_batch_static_all_1.csv')
    except:
        errors_baseModels_batch.append([input_width, label_width, model])


# In[ ]:


Batchstatic_ACT= pd.DataFrame(results_baseModels_batch)
Batchstatic_ACT.to_csv('Batchstatic_ACT.csv')

