from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy
import pandas as pd
import math
from sklearn.metrics import confusion_matrix, accuracy_score
import os

# setting a seed for reproducibility
numpy.random.seed(10)

'''
=============================================================================================
                read all stock files in directory called sl_stocks
'''


def read_all_Srilankan_stock_files(folder_path):
    all_csv_Files = []
    for (_, _, files) in os.walk(folder_path):
        all_csv_Files.extend(files)
        break

    dataframe_Stock_dict = {}
    for sl_stock_file in all_csv_Files:
        sl_stock_df = pd.read_csv(folder_path + "/" + sl_stock_file)
        dataframe_Stock_dict[(sl_stock_file.split('_'))[0]] = sl_stock_df

    return dataframe_Stock_dict


'''
=============================================================================================
         convert an array of values into a dataset matrix (create_stock_dataset)
'''


def create_stock_dataset(sl_dataset, val=1):
    stock_dataX, stock_dataY = [], []
    for i in range(len(sl_dataset) - val):
        x = sl_dataset[i:(i + val), 0]
        stock_dataX.append(x)
        stock_dataY.append(sl_dataset[i + val, 0])
    # print(stock_dataX)
    # print('-------------------------')
    # print(stock_dataY)
    return numpy.array(stock_dataX), numpy.array(stock_dataY)


'''
=============================================================================================
         create a proper stock dataset from the dataframe
'''


def create_sl_stock_preprocessed_Dataset(stock_df):
    stock_df.drop(stock_df.columns.difference(['date', 'open', 'close']), 1, inplace=True)
    stock_dfnew = stock_df
    stock_df = stock_df['open']
    stock_df1 = stock_dfnew['close']

    stock_dataset = stock_df.values
    stock_dataset1 = stock_df1.values

    stock_dataset = stock_dataset.reshape(-1, 1)
    stock_dataset1 = stock_dataset1.reshape(-1, 1)

    stock_dataset = stock_dataset.astype('float32')
    stock_dataset1 = stock_dataset1.astype('float32')

    # split data set into into training set and testing sets
    sl_train_size = len(stock_dataset) - 2
    sl_train, sl_test = stock_dataset[0:sl_train_size, :], stock_dataset[sl_train_size:len(stock_dataset), :]

    sl_train_size1 = len(stock_dataset1) - 2
    sl_train1, sl_test1 = stock_dataset1[0:sl_train_size1, :], stock_dataset1[sl_train_size1:len(stock_dataset1), :]

    # print('------------print sl train------------------------')
    # print(sl_train.shape)
    # print('---------------print sl test---------------------')
    # print(sl_test.shape)

    # reshape into X=t and Y=t+1
    val = 1
    sl_trainX, sl_trainY = create_stock_dataset(sl_train, val)
    sl_testX, sl_testY = create_stock_dataset(sl_test, val)

    sl_trainX_close, sl_trainY_close = create_stock_dataset(sl_train1, val)
    sl_testX_close, sl_testY_close = create_stock_dataset(sl_test1, val)
    # print('-----------------------------')
    # print()
    # print('----------------sl_trainX--------------------')
    # print(sl_trainX.shape)
    # print('----------------sl_trainY------------------')
    # print(sl_trainY.shape)
    # print('----------------sl_testX--------------------')
    # print(sl_testX.shape)
    # print('----------------sl_testY--------------------')
    # print(sl_testY.shape)

    return sl_trainX, sl_trainY, sl_testX, sl_testY, sl_trainX_close, sl_trainY_close, sl_testX_close, sl_testY_close


'''
=============================================================================================
          extract input dates and opening price value of stocks
'''


def get_sl_stock_Data(stock_df):
    # Create the lists / X and Y data sets
    dates = []
    sl_prices = []
    sl_close = []

    '''Get the last row of data (this will be the data that we test on)'''
    last_data_row = stock_df.tail(1)

    '''Get all of the data except for the last row'''
    stock_df = stock_df.head(len(stock_df) - 1)

    '''Get all of the rows from the Date Column'''
    df_dates = stock_df.loc[:, 'date']

    '''Get all of the rows from the Open Column'''
    df_open = stock_df.loc[:, 'open']

    '''Get all of the rows from the Open Column'''
    df_close = stock_df.loc[:, 'close']

    '''Create the independent data set X'''
    for sl_date in df_dates:
        dates.append([int(sl_date.split('-')[2])])

    '''Create the dependent data set 'y'''
    for open_price in df_open:
        sl_prices.append(float(open_price))

    '''Create the dependent data set 'y'''
    for close_price in df_open:
        sl_close.append(float(close_price))

    '''last row'''
    l_date = int(((list(last_data_row['date']))[0]).split('-')[2])
    l_price = float((list(last_data_row['open']))[0])
    l_close = float((list(last_data_row['close']))[0])

    return dates, sl_prices, l_date, l_price, sl_close, l_close


'''
=============================================================================================
                                 linear regression
'''


def linear_regression(dates, sl_prices, sl_test_date, sl_df):
    linear_reg = LinearRegression()

    sl_trainX, sl_trainY, sl_testX, sl_testY, sl_trainX_close, sl_trainY_close, sl_testX_close, sl_testY_close = create_sl_stock_preprocessed_Dataset(
        sl_df)

    X_train, X_test, y_train, y_test = train_test_split(sl_trainX, sl_trainY, test_size=0.33, random_state=42)
    X_train_close, X_test_close, y_train_close, y_test_close = train_test_split(sl_trainX_close, sl_trainY_close,
                                                                                test_size=0.33, random_state=42)

    linear_reg.fit(sl_trainX, sl_trainY)
    linear_reg.fit(sl_trainX_close, sl_trainY_close)

    predict_decision_boundary = linear_reg.predict(sl_trainX)
    predict_decision_boundary_close = linear_reg.predict(sl_trainX_close)

    linear_reg_y_pred = linear_reg.predict(X_test)
    linear_reg_y_pred_close = linear_reg.predict(X_test_close)

    mean_squared_error_test_score = mean_squared_error(y_test, linear_reg_y_pred)
    mean_squared_error_test_score_close = mean_squared_error(y_test_close, linear_reg_y_pred_close)

    prediction_of_linear_reg = linear_reg.predict(sl_testX)[0]
    prediction_of_linear_reg_close = linear_reg.predict(sl_testX_close)[0]

    return predict_decision_boundary, prediction_of_linear_reg, mean_squared_error_test_score, prediction_of_linear_reg_close


'''
=============================================================================================
                                         SVM 
'''


def SVR_rbf(dates, prices, test_date, sl_df):
    svm = SVR(kernel='rbf', C=1e3, gamma=0.1)

    sl_trainX, sl_trainY, sl_testX, sl_testY, sl_trainX_close, sl_trainY_close, sl_testX_close, sl_testY_close = create_sl_stock_preprocessed_Dataset(
        sl_df)

    X_train, X_test, y_train, y_test = train_test_split(sl_trainX, sl_trainY, test_size=0.33, random_state=42)
    X_train_close, X_test_close, y_train_close, y_test_close = train_test_split(sl_trainX_close, sl_trainY_close,
                                                                                test_size=0.33, random_state=42)

    svm.fit(sl_trainX, sl_trainY)
    svm.fit(sl_trainX_close, sl_trainY_close)

    svm_decision_boundary = svm.predict(sl_trainX)
    predict_decision_boundary_close = svm.predict(sl_trainX_close)

    svm_y_pred = svm.predict(X_test)
    svm_reg_y_pred_close = svm.predict(X_test_close)

    svm_test_score = mean_squared_error(y_test, svm_y_pred)
    mean_squared_error_test_score_close = mean_squared_error(y_test_close, svm_reg_y_pred_close)

    svm_prediction = svm.predict(sl_testX)[0]
    prediction_of_svm_close = svm.predict(sl_testX_close)[0]

    return svm_decision_boundary, svm_prediction, svm_test_score, prediction_of_svm_close


'''
=============================================================================================
                                    random forests
'''


def random_forests(dates, sl_prices, sl_test_date, sl_df):
    randomForest = RandomForestRegressor(n_estimators=10, random_state=0)

    sl_trainX, sl_trainY, sl_testX, sl_testY, sl_trainX_close, sl_trainY_close, sl_testX_close, sl_testY_close = create_sl_stock_preprocessed_Dataset(
        sl_df)

    X_train, X_test, y_train, y_test = train_test_split(sl_trainX, sl_trainY, test_size=0.33, random_state=42)
    X_train_close, X_test_close, y_train_close, y_test_close = train_test_split(sl_trainX_close, sl_trainY_close,
                                                                                test_size=0.33, random_state=42)

    randomForest.fit(sl_trainX, sl_trainY)
    randomForest.fit(sl_trainX_close, sl_trainY_close)


    predict_decision_boundary = randomForest.predict(sl_trainX)
    predict_decision_boundary_close = randomForest.predict(sl_trainX_close)

    randomForest_y_pred = randomForest.predict(X_test)
    randomForest_y_pred_close = randomForest.predict(X_test_close)

    mean_squared_error_test_score = mean_squared_error(y_test, randomForest_y_pred)
    mean_squared_error_test_score_close = mean_squared_error(y_test_close, randomForest_y_pred_close)

    prediction_of_randomForest = randomForest.predict(sl_testX)[0]
    prediction_of_randomForest_close = randomForest.predict(sl_testX_close)[0]

    return predict_decision_boundary, prediction_of_randomForest, mean_squared_error_test_score,prediction_of_randomForest_close


'''
=============================================================================================
                                        KNN 
'''


def KNN(dates, sl_prices, sl_test_date, sl_df):
    knn = KNeighborsRegressor(n_neighbors=2)

    sl_trainX, sl_trainY, sl_testX, sl_testY, sl_trainX_close, sl_trainY_close, sl_testX_close, sl_testY_close = create_sl_stock_preprocessed_Dataset(
        sl_df)

    X_train, X_test, y_train, y_test = train_test_split(sl_trainX, sl_trainY, test_size=0.33, random_state=42)
    X_train_close, X_test_close, y_train_close, y_test_close = train_test_split(sl_trainX_close, sl_trainY_close,
                                                                                test_size=0.33, random_state=42)

    knn.fit(sl_trainX, sl_trainY)
    knn.fit(sl_trainX_close, sl_trainY_close)

    knn_decision_boundary = knn.predict(sl_trainX)
    predict_decision_boundary_close = knn.predict(sl_trainX_close)

    knn_y_pred = knn.predict(X_test)
    knn_y_pred_close = knn.predict(X_test_close)

    knn_test_score = mean_squared_error(y_test, knn_y_pred)
    mean_squared_error_test_score_close = mean_squared_error(y_test_close, knn_y_pred_close)

    knn_prediction = knn.predict(sl_testX)[0]
    prediction_of_knn_close = knn.predict(sl_testX_close)[0]

    # knn_accuracy_score = accuracy_score(y_test, knn_y_pred)

    return knn_decision_boundary, knn_prediction, knn_test_score,prediction_of_knn_close


'''
=============================================================================================
                                        decision trees 
'''


def DT(dates, sl_prices, sl_test_date, sl_df):
    decision_trees = tree.DecisionTreeRegressor()

    sl_trainX, sl_trainY, sl_testX, sl_testY, sl_trainX_close, sl_trainY_close, sl_testX_close, sl_testY_close = create_sl_stock_preprocessed_Dataset(
        sl_df)

    X_train, X_test, y_train, y_test = train_test_split(sl_trainX, sl_trainY, test_size=0.33, random_state=42)
    X_train_close, X_test_close, y_train_close, y_test_close = train_test_split(sl_trainX_close, sl_trainY_close,
                                                                                test_size=0.33, random_state=42)

    decision_trees.fit(sl_trainX, sl_trainY)
    decision_trees.fit(sl_trainX_close, sl_trainY_close)

    decision_boundary = decision_trees.predict(sl_trainX)
    predict_decision_boundary_close = decision_trees.predict(sl_trainX_close)

    decision_trees_y_pred = decision_trees.predict(X_test)
    decision_trees_y_pred_close = decision_trees.predict(X_test_close)

    decision_trees_test_score = mean_squared_error(y_test, decision_trees_y_pred)
    mean_squared_error_test_score_close = mean_squared_error(y_test_close, decision_trees_y_pred_close)

    decision_trees_prediction = decision_trees.predict(sl_testX)[0]
    prediction_of_knn_close = decision_trees.predict(sl_testX_close)[0]

    return decision_boundary, decision_trees_prediction, decision_trees_test_score,prediction_of_knn_close
