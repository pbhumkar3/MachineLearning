# course: TCSS555
# ML in Python, homework 3
# date: 05/09/2018
# name: Martine De Cock
# updated by : Pradnya Bhumkar
# description: Neural network for predicting personality of Facebook users

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from keras import optimizers

import numpy as np
import pandas as pd

# Fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Loading the data
# There are 9500 users (rows)
# There are 81 columns for the LIWC features followed by columns for
# openness, conscientiousness, extraversion, agreeableness, neuroticism
# As the target variable, we select the extraversion column (column 83)


def read_data_as_matrix():
    dataset = np.loadtxt("/Users/abhi/c:temp/tcss555/training/testfile.csv", delimiter=",")
    X = dataset[:, 1:84]
    y = dataset[:, 87]  # agr
    return X, y


def neural_network_openess(dataset):
    X = dataset[:, 1:84]
    y = dataset[:, 87]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1500)

    model = Sequential()

    model.add(Dense(83, input_dim=83, kernel_initializer='normal', activation='sigmoid'))
    model.add(Dense(40, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(optimizer='Adamax', loss='mse', metrics=['mse'])
    model.fit(X_train, y_train, epochs=10)
    y_pred = model.predict(X_test)

    print('MSE with neural network for ope:', metrics.mean_squared_error(y_test, y_pred))
    print('RMSE with neural network for ope:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))



def neural_network_ext(dataset):
    X = dataset[:, 1:84]
    y = dataset[:, 90]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1500)

    model = Sequential()

    model.add(Dense(83, input_dim=83, kernel_initializer='normal', activation='sigmoid'))
    model.add(Dense(40, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(optimizer='Adamax', loss='mse', metrics=['mse'])
    model.fit(X_train, y_train, epochs=10)
    y_pred = model.predict(X_test)

    print('MSE with neural network for ext:', metrics.mean_squared_error(y_test, y_pred))
    print('RMSE with neural network for ext:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


def neural_network_nue(dataset):
    X = dataset[:, 1:84]
    y = dataset[:, 88]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1500)

    model = Sequential()

    model.add(Dense(83, input_dim=83, kernel_initializer='normal', activation='sigmoid'))
    model.add(Dense(40, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(optimizer='Adamax', loss='mse', metrics=['mse'])
    model.fit(X_train, y_train, epochs=10)
    y_pred = model.predict(X_test)

    print('MSE with neural network for nue:', metrics.mean_squared_error(y_test, y_pred))
    print('RMSE with neural network for nue:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


def neural_network_agr(dataset):
    X = dataset[:, 1:84]
    y = dataset[:, 86]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1500)

    model = Sequential()

    model.add(Dense(83, input_dim=83, kernel_initializer='normal', activation='sigmoid'))
    model.add(Dense(40, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(optimizer='Adamax', loss='mse', metrics=['mse'])
    model.fit(X_train, y_train, epochs=10)
    y_pred = model.predict(X_test)

    print('MSE with neural network for agr:', metrics.mean_squared_error(y_test, y_pred))
    print('RMSE with neural network for agr:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


def neural_network_con(dataset):
    X = dataset[:, 1:84]
    y = dataset[:, 89]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1500)

    model = Sequential()

    model.add(Dense(83, input_dim=83, kernel_initializer='normal', activation='sigmoid'))
    model.add(Dense(40, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(optimizer='Adamax', loss='mse', metrics=['mse'])
    model.fit(X_train, y_train, epochs=10)
    y_pred = model.predict(X_test)

    print('MSE with neural network for con:', metrics.mean_squared_error(y_test, y_pred))
    print('RMSE with neural network for con:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


def main():
    output_dir = "/Users/abhi/c:temp/tcss555/public-test-data/output_folder"
    train_liwc_file = '/Users/abhi/c:temp/tcss555/training/LIWC/LIWC.csv'
    train_profile_file = '/Users/abhi/c:temp/tcss555/training/profile/profile.csv'

    test_profile_file = '/Users/abhi/c:temp/tcss555/public-test-data/profile/profile.csv'
    test_liwc_file = '/Users/abhi/c:temp/tcss555/public-test-data/LIWC/LIWC.csv'

    profile_train_data = pd.read_csv(train_profile_file)
    profile_test_data = pd.read_csv(test_profile_file)
    liwc_train_data = pd.read_csv(train_liwc_file)
    test_LIWC_data = pd.read_csv(test_liwc_file)

    training_data = pd.merge(left=liwc_train_data, right=profile_train_data, how='left', left_on='userId', right_on='userid')
    training_data = training_data.drop('userId', axis=1)
    training_data = training_data.drop('userid', axis=1)
    training_data.to_csv('/Users/abhi/c:temp/tcss555/training/testfile.csv', sep=',', encoding='utf-8', header = False)

    dataset = np.loadtxt("/Users/abhi/c:temp/tcss555/training/testfile.csv", delimiter=",")

    neural_network_openess(dataset)
    neural_network_agr(dataset)
    neural_network_con(dataset)
    neural_network_ext(dataset)
    neural_network_nue(dataset)



if __name__ == "__main__":
    #args = parse_arguments()
    main()