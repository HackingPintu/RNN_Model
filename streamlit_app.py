import pandas as pd
df1 = pd.read_excel('newmoli.xlsx')
df = df1.copy()
df['date_of_job_post'] = pd.to_datetime(df['date_of_job_post'], format='%d-%m-%Y %H:%M:%S')
df['application_date'] = pd.to_datetime(df['application_date'], format='%d-%m-%Y %H:%M:%S')

df = df[(df['time_difference_minutes'] >= 1) & (df['time_difference_minutes'] <= 750)]
 
df['job_post_hour'] = df['date_of_job_post'].dt.hour
df['job_post_minutes'] = df['date_of_job_post'].dt.minute
df['job_post_seconds'] = df['date_of_job_post'].dt.second
df['job_post_day'] = df['date_of_job_post'].dt.dayofweek
df['job_post_month'] = df['date_of_job_post'].dt.month
df['job_post_year'] = df['date_of_job_post'].dt.year
 
df['application_hour'] = df['application_date'].dt.hour
df['application_minutes'] = df['application_date'].dt.minute
df['application_seconds'] = df['application_date'].dt.second
df['application_day'] = df['application_date'].dt.dayofweek
df['application_month'] = df['application_date'].dt.month
df['application_year'] = df['application_date'].dt.year
 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
 
X = df[['job_id', 'entity_id', 'job_post_hour','job_post_minutes', 'job_post_day','job_post_month','job_post_year','application_hour','application_minutes' ,'application_day','application_month','application_year', 'rate', 'certification']]
y = df['time_difference_minutes']
 
X = pd.get_dummies(X, columns=['job_post_day', 'job_post_month', 'application_day', 'application_month'])
 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
 
 
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=56)
 
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
 
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(75, activation='relu', return_sequences=True))
model.add(LSTM(100, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train, y_train, epochs=400, batch_size=32, validation_split=0.2, verbose=0)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
  
print("R-squared Score:", r2)

import streamlit as st
st.success(f"{r2}")
