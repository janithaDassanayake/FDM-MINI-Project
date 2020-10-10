from flask import Flask, render_template, request, send_from_directory
import utils
import train_models as train_model
import math
import numpy as np
import pandas as pd

app = Flask(__name__)


def perform_training(sl_stock_name, stock_df, predict_models_list, forcastingDays):
    plotting_colors = {
        'SVR_rbf': '#FFA646',
        'linear_regression': '#CC2A1E',
        'random_forests': '#8F0099',
        'KNN': '#CCAB43',
        'DT': '#85CC43'}

    dates, stock_prices, prediction_models_outputs, prediction_date, test_price, close_prices, l_close_price = train_model.train_predict_plot(
        sl_stock_name, stock_df, predict_models_list, forcastingDays)

    original_dates = dates
    #
    # print(dates)
    # print('\n')
    # print(stock_prices)
    # print('\n')
    # print(prediction_models_outputs)
    # print('\n')
    # print(prediction_date)
    # print('\n')
    # print(test_price)

    if len(dates) > 40:
        dates = dates[-40:]
        sl_stock_prices = stock_prices[-40:]

    all_data_list = []
    all_data_list.append((sl_stock_prices, 'false', 'Actual price', '#000000'))

    # print(all_data_list)

    # print()
    ''' {'random_forests': (array([14.91800032, 14.57600002, 14.40400009, ..., 54.,
                                      53.47440155, 52.50700073]), 48.99099845886231, 0.34788879363153224)}'''
    # print('\n')prediction_models_outputs

    for model_name in prediction_models_outputs:
        # print(model_name)
        # print('\n')
        if len(original_dates) > 40:
            all_data_list.append(
                (((prediction_models_outputs[model_name])[0])[-40:], "true", model_name, plotting_colors[model_name]))
            # print((prediction_models_outputs[model_name][0]))
            '''[15.29382304 15.08818794 14.52529572 ... 52.97565052 53.70898119 53.27664601]'''
        else:
            all_data_list.append(
                (((prediction_models_outputs[model_name])[0]), "true", model_name, plotting_colors[model_name]))

    # print(all_data_list)
    '''(array([50.265   , 50.35    , 51.39    , 50.54    , 51.09    , 49.545   ,
       52.75    , 53.36    , 53.57    , 52.83    ], dtype=float32), 'true', 'KNN', '#CCAB43')]
    '''

    all_stock_prediction_data = []
    all_stock_test_evaluations = []
    all_stock_close_price_prediction = []
    list_of_close_prices = []
    list1 = []
    forcast_date_List_new = []

    all_stock_prediction_data.append(("Original value", test_price))
    all_stock_close_price_prediction.append(("Original close value", l_close_price))

    for model_name in prediction_models_outputs:

        ''' append calculated predicted opening values to all_stock_prediction_data list '''
        all_stock_prediction_data.append((model_name, (prediction_models_outputs[model_name])[1]))

        ''' append calculated mean_squared_error values to all_stock_prediction_data list '''
        all_stock_test_evaluations.append((model_name, (prediction_models_outputs[model_name])[2]))

        ''' append calculated close price values to all_stock_prediction_data list '''
        all_stock_close_price_prediction.append((model_name, (prediction_models_outputs[model_name])[3]))

        list_of_close_prices.append(prediction_models_outputs[model_name][3])

        forcast_price_list = prediction_models_outputs[model_name][4]

        for x in range(0, int(forcastingDays)):
            roundforcast=forcast_price_list[x]
            roundforcast=round(roundforcast,4)
            list1.append(roundforcast)
        print(list1)

    list_of_close_prices.append(l_close_price)

    list_of_close_prices.extend(list1)
    print('this is best ', list_of_close_prices)

    maxprice = max(list_of_close_prices)
    minprice = min(list_of_close_prices)

    minprice = math.floor(minprice)
    maxprice = math.ceil(maxprice)

    startdate = prediction_date
    for x in range(1, int(forcastingDays) + 1):
        enddate = pd.to_datetime(startdate) + pd.DateOffset(days=x)
        enddate = str(enddate)
        input = enddate.replace(' 00:00:00', '')
        forcast_date_List_new.append(input)

    return all_stock_prediction_data, all_stock_prediction_data, prediction_date, dates, all_data_list, all_data_list, all_stock_test_evaluations, all_stock_close_price_prediction, maxprice, minprice, forcast_date_List_new, list1


all_stock_files = utils.read_all_Srilankan_stock_files('sl_stock_files')

'''============================================================================================================'''


@app.route('/')
def landing_function():
    sl_stock_files = list(all_stock_files.keys())
    '''['AAL', 'AAPL', 'AAP', 'ABBV', 'ABC', 'ABT', 'ACN', 'ADBE', 'ADI'],...........'''

    return render_template('index.html', show_results_output="false", stock_len=len(sl_stock_files),
                           sl_stock_files=sl_stock_files,
                           len_2=len([]),
                           all_prediction_stock_data=[],
                           prediction_result_date="", dates=[], maxprice="", minprice="", all_stock_data=[],
                           all_close_price=[], forcasting=[], forcastingdate=[], len3=len([]),
                           len=len([]))


@app.route('/process', methods=['POST'])
def process():
    # get the selected stock file
    sl_stock_file_name = request.form['stock_file_name']
    # get the selected prediction models list
    Prediction_Model_algoritms = request.form.getlist('Prediction_Model')

    forcastingDays = request.form['H_id']

    df = all_stock_files[str(sl_stock_file_name)]

    all_prediction_data, all_prediction_data, prediction_date, dates, all_data, all_data, all_test_evaluations, all_stock_close_price_prediction1, maxprice, minprice, forcast_date_List_new, list1 = perform_training(
        str(sl_stock_file_name), df, Prediction_Model_algoritms, forcastingDays)

    stock_files = list(all_stock_files.keys())

    return render_template('index.html', all_test_evaluations=all_test_evaluations, show_results_output="true",
                           stock_len=len(stock_files), sl_stock_files=stock_files,
                           len_2=len(all_prediction_data),
                           all_prediction_stock_data=all_prediction_data,
                           prediction_result_date=prediction_date, dates=dates, all_stock_data=all_data,
                           all_close_price=all_stock_close_price_prediction1, min_price=minprice, max_price=maxprice,
                           forcasting=list1, forcastingdate=forcast_date_List_new, len3=len(forcast_date_List_new),
                           len=len(all_data))


# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    app.run()
