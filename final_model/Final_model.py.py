#!/usr/bin/env python
# coding: utf-8

# 
# # Model LSTM and RNN 
# New Try with log of our target


from datetime import datetime
import random
import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler, MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer, Masking
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, regularizers
from keras.callbacks import EarlyStopping



def get_X_y(df, 
            X_length, 
            y_length, 
            number_of_sequences=51, 
            number_of_targets=1, 
            val=False, 
            val_cutoff=0.8):


#     limit dataframes to length for train/test splits
    df_X = df.copy()
    df_y = df.iloc[:, :number_of_targets].copy()
    
#     convert and scale X dataframe to PCA to solve dimensionality problem
    scaler = MinMaxScaler()
    df_X_scaled = pd.DataFrame(scaler.fit_transform(df_X))
    
#     create unique list to sample random datapoints from
    if val:
        sample_list = list(range(int(len(df_y)*val_cutoff), int(len(df_y)-y_length)))
    if not val:
        sample_list = list(range(int(X_length), int(len(df_y)-y_length)))
    random.shuffle(sample_list)
    
#     empty lists to append data to, will create 3D dataframe here
    X, y = [], []
    
    
    
#     define a simple data slicing and selection function. This function will create a slice of data from a specified random starting position. The random position must be generated externally.
    
    def get_Xi_yi(df_X, 
              df_y,
              random_start,
              X_length, 
              y_length):
    
#     must define a random_start:int for function to run
        Xi = df_X.iloc[random_start-X_length:random_start]
        yi = df_y.iloc[random_start:random_start+y_length]

        return Xi, yi

    
#     for loop to select ith values from data
    for i in range(number_of_sequences):
        Xi, yi = get_Xi_yi(df_X_scaled, df_y, sample_list.pop(), X_length, y_length)
        X.append(Xi.values.tolist())
        y.append(yi.values.tolist())
        
    return np.array(X), np.array(y)



def load_data(date_start, date_end):
    url = "https://raw.githubusercontent.com/Tobias-Neubert94/adam_monk_II/master/adam_monk_II/data/Price_data_new_2.gzip"
    df = pd.read_parquet(url)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df = df.loc[date_start:date_end]
    
    
    #df = pd.read_parquet("df.gzip")
    #df['date'] = pd.to_datetime(df['date'])
    #df = df.set_index('date')
    
    
    df = df[['future_price'] + [col for col in df.columns if col != 'future_price']]
    #try to do the log
    epsilon = 1e-8  # there are some 0 Values in the dataframe. Cannot work then
    #df['future_price'] = np.log(df['future_price'] + epsilon)
    #df = df[df.index.year < 2020]
    return df



start_date = pd.to_datetime('2003-01-01')
end_date = pd.to_datetime('2020-12-31')
df = load_data(start_date, end_date)


#split_train_test_set
points_to_predict = 7
X_train, y_train = get_X_y(df.iloc[:int(len(df)*0.9)], 
                           X_length=56, 
                           y_length=points_to_predict, 
                           number_of_sequences=365*3, 
                           number_of_targets=1)
y_train = y_train[:, :, 0]


X_val, y_val = get_X_y(df.iloc[:int(len(df)*0.99)], 
                       X_length=56, 
                       y_length=points_to_predict, 
                       number_of_sequences=100, 
                       number_of_targets=1, 
                       val=True, 
                       val_cutoff=0.9)
y_val = y_val[:, :, 0]

X_test, y_test = get_X_y(df, 
                         X_length=56, 
                         y_length=points_to_predict, 
                         number_of_sequences=10, 
                         number_of_targets=1, 
                         val=True, 
                         val_cutoff=0.99)
y_test = y_test[:, :, 0]


#target Scaling
# y normalize
epsilon = 1e-8  # there are some 0 Values in the dataframe. Cannot work then
y_scaler = MinMaxScaler()
y_scaler.fit(np.log(y_train+ epsilon))

y_train_scaled = y_scaler.transform(np.log(y_train+ epsilon))
y_test_scaled = y_scaler.transform(np.log(y_test+ epsilon))
y_val_scaled = y_scaler.transform(np.log(y_val+ epsilon))


#building model
def build_model():
    input_shape = X_train.shape[1:]
    output_shape = y_train_scaled.shape[1]
    
    print(input_shape)
    print(output_shape)
    #reg = regularizers.l1_l2(l1=0.04, l2=0.02)

    model_LSTM = Sequential()
    model_LSTM.add(LSTM(units=200, return_sequences=True, input_shape=input_shape))
    model_LSTM.add(LSTM(units=150, return_sequences=True))
    model_LSTM.add(LSTM(units=150, return_sequences=True))
    model_LSTM.add(LSTM(units=150, return_sequences=False))
    model_LSTM.add(Dense(units=output_shape, activation='linear'))
    
    model_LSTM.summary()
    return model_LSTM
    
model_LSTM = build_model()

#fit
def train_model(optimizer, batch_size, epochs):
    #import tensorflow_addons as tfa ##not needed yet
    model_LSTM.compile(
        loss=['mae'], 
        metrics=['mae'],
        #optimizer=tfa.optimizers.AdamW(learning_rate=0.00003, beta_1=0.8, epsilon=1e-18, weight_decay=0.01)
        optimizer = optimizer
    )
    
    es = EarlyStopping(
        patience=5, 
        restore_best_weights=True
    )
    
    history = model_LSTM.fit(
                    X_train, 
                    y_train_scaled, 
                    batch_size = batch_size, 
                    epochs = epochs,
                    shuffle = False,
                    verbose = 2,
                    validation_data = (X_val, y_val_scaled), 
                    #callbacks=[es]
    )
    
    y_pred = model_LSTM.predict(X_test)
    
    y_pred = y_scaler.inverse_transform(y_pred)
    y_pred = np.exp(y_pred)
    return y_pred, y_test, history
    
y_pred, y_test, history = train_model('nadam', 16, 30 )

###I dont know if its needed
for i in range(y_pred.shape[0]):
    fig = plt.gcf(); fig.set_size_inches(20, 12);
    plt.plot(y_pred[i, :], c="blue", label="Prediction")
    plt.plot(y_test[i, :], c="red", label="Test")
#     plt.ylim(0, 430)
    plt.legend(loc='upper right')
    plt.show();

dl_error = np.mean(abs(y_pred - y_test))

base_error = np.mean(abs((df[['future_price']]-df[['future_price']].shift(7)).dropna()))[0]


def plot_history(history):
    
    fig, ax = plt.subplots(1,2, figsize=(20,7))
    # --- LOSS: LOSS --- 
    ax[0].plot(history.history['loss'])
    ax[0].plot(history.history['val_loss'])
    ax[0].set_title('Loss')
    ax[0].set_ylabel('LOSS')
    ax[0].set_xlabel('EPOCH')
    ax[0].legend(['Train', 'Validation'], loc='best')
    ax[0].grid(axis="x",linewidth=0.5)
    ax[0].grid(axis="y",linewidth=0.5)
    
    # --- METRICS: MAE ---
    
    ax[1].plot(history.history['mae'])
    ax[1].plot(history.history['val_mae'])
    ax[1].set_title('Mean Absolute Error')
    ax[1].set_ylabel('LOSS')
    ax[1].set_xlabel('EPOCH')
    ax[1].legend(['Train', 'Validation'], loc='best')
    ax[1].grid(axis="x",linewidth=0.5)
    ax[1].grid(axis="y",linewidth=0.5)
                        
    return ax

plot_history(history);

#To Do:
    #Option: how many days the user wants to predict
    #exponent
    #analyse pred data ==> columsn ==> exact date
    #Compare predict vs. actual
    #insert "old" model ==> obsolete since we only want to show the productive model



predict_dates = df.index[-len(y_test)]

results = []

for i in range(len(y_pred)):
    sample_results = []
    for j in range(len(y_pred[i])):
        date = predict_dates + pd.DateOffset(days=j)
        predicted_value = y_pred[i][j]
        real_value = y_test[i][j]
        sample_results.append([date, predicted_value, real_value])
    results.append(sample_results)

columns = ['Date', 'Predicted Value', 'Real Value']
df_results = pd.DataFrame(results[0], columns=columns)
df_results.set_index('Date', inplace=True)