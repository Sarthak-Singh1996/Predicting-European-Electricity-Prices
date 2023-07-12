import streamlit as st
from streamlit import config
from datetime import datetime
import random
import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler, MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns

from keras.optimizers import Nadam
from keras.layers import LSTM, Dense, Dropout, InputLayer, Masking
from keras.models import Sequential
from keras import layers, regularizers
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import save_model, load_model
import pickle
import urllib.request

st.set_page_config(layout="wide")

st.title("Electricty Future Price Prediction App")
start_button = st.button("Start")
stop_button = st.button("Stop")
st.markdown("Description: This app gives the prediction of future closed price of electricity for Germany.")

### Change sidebar color

st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: blue;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("Predictor Settings")

# Create the start and stop buttons



st.sidebar.subheader("Model Parameters Selection:")

START_DATE = datetime(2003, 1, 1)
END_DATE = datetime(2022, 12, 31)
st.sidebar.subheader("Time Frame Selection:")
selected_start_date = st.sidebar.date_input("Start Date", value=START_DATE, min_value=START_DATE, max_value=END_DATE)
selected_end_date = st.sidebar.date_input("End Date", value=END_DATE, min_value=START_DATE, max_value=END_DATE)

# Display the selected time frame

st.write("Start Date:", selected_start_date)
st.write("End Date:", selected_end_date)

#Functions:
##TN fix variable
periods = 7
#Split the data inside the frame
def get_X_y(df,
            X_length=56,
            y_length=periods,
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


# Load data

def load_data(date_start, date_end):
    url = "https://raw.githubusercontent.com/Tobias-Neubert94/adam_monk_II/master/adam_monk_II/data/Final_data.gzip"
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

df = load_data(selected_start_date, selected_end_date) # dataset issues when it comes to choose the start_date 2013 sept and onwards, also it's better if we show the data as raw rather doing log


# split_train_test_set

X_train, y_train = get_X_y(df.iloc[:int(len(df)*0.9)],
                           X_length=56,
                           y_length=periods,
                           number_of_sequences=365*3,
                           number_of_targets=1)
y_train = y_train[:, :, 0]


X_val, y_val = get_X_y(df.iloc[:int(len(df)*0.99)],
                       X_length=56,
                       y_length=periods,
                       number_of_sequences=100,
                       number_of_targets=1,
                       val=True,
                       val_cutoff=0.9)
y_val = y_val[:, :, 0]

X_test, y_test = get_X_y(df,
                         X_length=56,
                         y_length=periods,
                         number_of_sequences=10,
                         number_of_targets=1,
                         val=True,
                         val_cutoff=0.99)
y_test = y_test[:, :, 0]


def load_LSTM_model_func(model_url, model_filename, history_filename=None):
    urllib.request.urlretrieve(model_url, model_filename)
    model = load_model(model_filename)
    
    if history_filename is not None:
        with open(history_filename, "rb") as file:
            history = pickle.load(file)
        return model, history
    
    return model

#fit
def predict_model():
    #target Scaling
    # y normalize
    epsilon = 1e-8  # there are some 0 Values in the dataframe. Cannot work then
    y_scaler = MinMaxScaler()
    y_scaler.fit(np.log(y_train+ epsilon))
    model_LSTM, history = load_LSTM_model_func("https://raw.githubusercontent.com/Tobias-Neubert94/adam_monk_II/master/adam_monk_II/data/trained_model.h5", "trained_model.h5", history_filename="training_history.pickle")

    y_pred = model_LSTM.predict(X_test)

    y_pred = y_scaler.inverse_transform(y_pred)
    y_pred = np.exp(y_pred)
    return y_pred, y_test, history


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

    return fig

#Predicted value

def show_prediction():
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

    columns = ['Date', 'Predicted Value', 'Actual Value']
    df_results = pd.DataFrame(results[0], columns=columns)
    # df_results.set_index('Date', inplace=True) #check it
    return df_results

# Generate plot
def generate_plot():
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

    columns = ['Date', 'Predicted Value', 'Actual Value']
    df_results = pd.DataFrame(results[0], columns=columns)
    # df_results.set_index('Date', inplace=True) # check it
    # Create a figure and axes for the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the actual data
    ax.plot(df_results['Date'], df_results['Actual Value'], label='Actual')

    # Plot the predicted data
    ax.plot(df_results['Date'], df_results['Predicted Value'], label='Prediction')

    # Add a legend
    ax.legend()

    # Return the figure
    return fig


if start_button:

    #load data
    data_load_state = st.sidebar.text("Load data....")
    data_load_state.text("Loading data...done!")
    st.subheader('Raw data')
    st.write(df.head())

    y_pred,y_test,history = predict_model()

    # Predicted data
    prediction = show_prediction()
    st.subheader("Predicted data vs Actual data")
    st.write(prediction)

    # Generate plot
    fig = generate_plot()
    st.subheader("Prediction vs Actual")
    st.plotly_chart(fig)

    # Plot history
    fig = plot_history(history)
    st.subheader("Loss and MAE over epochs")
    st.plotly_chart(fig)
    # st.pyplot(fig)
    # Actual vs Prediction

    # Check if really required
    # for i in range(y_pred.shape[0]):
    #     fig = plt.gcf(); fig.set_size_inches(20, 12);
    #     plt.plot(y_pred[i, :], c="blue", label="Prediction")
    #     plt.plot(y_test[i, :], c="red", label="Test")
    #     # plt.ylim(0, 430)
    #     plt.legend(loc='upper right')
    #     plt.show();
    #     dl_error = np.mean(abs(y_pred - y_test))
    #     base_error = np.mean(abs((df[['future_price']]-df[['future_price']].shift(7)).dropna()))[0]

if stop_button:
    st.stop()
