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


st.set_page_config(layout="wide")

st.title("Electricty Future Price Prediction App")
start_button = st.button("Start")
stop_button = st.button("Stop")
st.markdown("Description of app")

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

# Select the desired optimizer

optimizer = ("Nadam","Adam","RMSPROP","ADAgrad","Adadelta","Adamax","SGD")
selected_optimizer = st.sidebar.selectbox("Select desired optimizer:",optimizer)

# Create number of days to predict

periods = int(st.sidebar.number_input('Number of days to predict:', min_value=0, max_value=1000, value=7, step=1))

# Select number of batchs and epochs

batchs = int(st.sidebar.number_input('Select the batche size:', min_value=0, max_value=1000, value=16, step=1))
epochs = int(st.sidebar.number_input('Select number of epochs:', min_value=0, max_value=1000, value=10, step=1))

# Create a date range user input

START_DATE = datetime(2003, 1, 1)
END_DATE = datetime(2022, 12, 31)
st.sidebar.subheader("Time Frame Selection:")
selected_start_date = st.sidebar.date_input("Start Date", value=START_DATE, min_value=START_DATE, max_value=END_DATE)
selected_end_date = st.sidebar.date_input("End Date", value=END_DATE, min_value=START_DATE, max_value=END_DATE)

# Display the selected time frame

st.write("Start Date:", selected_start_date)
st.write("End Date:", selected_end_date)

#Functions:

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
    url = "https://raw.githubusercontent.com/Tobias-Neubert94/adam_monk_II/master/adam_monk_II/data/Price_data_new.gzip"
    df = pd.read_parquet(url)
    df = df.loc[date_start:date_end]

    df = df[['future_price'] + [col for col in df.columns if col != 'future_price']]
    #try to do the log
    epsilon = 1e-8  # there are some 0 Values in the dataframe. Cannot work then
    df['future_price'] = np.log(df['future_price'] + epsilon) #let's not show the data in log,raw data we will show
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
def train_model(selected_optimizer,batchs,epochs):
    #import tensorflow_addons as tfa ##not needed yet
    model_LSTM.compile(
        loss=['mae'],
        metrics=['mae'],
        #optimizer=tfa.optimizers.AdamW(learning_rate=0.00003, beta_1=0.8, epsilon=1e-18, weight_decay=0.01)
        optimizer = selected_optimizer
    )

    es = EarlyStopping(
        patience=5,
        restore_best_weights=True
    )

    history = model_LSTM.fit(
                    X_train,
                    y_train_scaled,
                    batch_size = batchs,
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



if start_button:

    #load data
    data_load_state = st.sidebar.text("Load data....")
    data_load_state.text("Loading data...done!")
    st.subheader('Raw data')
    st.write(df.head())

    # Predicted data

    y_pred,y_test,history = train_model(selected_optimizer,batchs,epochs)
    st.subheader("Predicted data")
    st.write(y_pred)

    # Plot history
    fig = plot_history(history)
    st.subheader("Loss and MAE over epochs")
    st.plotly_chart(fig)

    # Actual vs Prediction
    st.subheader("Actual vs Prediction")

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
