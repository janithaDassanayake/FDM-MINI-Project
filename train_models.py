import matplotlib.pyplot as plt
import utils
import numpy

'''
=============================================================================================
                                         create plot 
'''


def create_plot(dates, sl_original_prices, prediction_models_outputs):
    plt.scatter(dates, sl_original_prices, color='red', label='blue')
    for model in prediction_models_outputs.keys():
        plt.plot(dates, (prediction_models_outputs[model])[0], color=numpy.random.rand(3, ), label=model)

    plt.xlabel('Days')
    plt.ylabel('stock Price')
    plt.title('Regression model')
    plt.legend()
    plt.savefig("train_Plot.png")
    plt.show()


def train_predict_plot(file_name, stock_df, prediction_ml_models_list,forcastingDays):
    prediction_models_outputs = {}

    dates, prices, test_date, test_stock_price, close_prices, l_close_price = utils.get_sl_stock_Data(stock_df)

    for prediction_model in prediction_ml_models_list:
        method_to_call = getattr(utils, prediction_model)
        prediction_models_outputs[prediction_model] = method_to_call(dates, prices, test_date, stock_df,forcastingDays)
    ''''{KNN': (array([15.030001, 14.795 , 14.5, ..., 53.3], dtype=float32), 49.01, 0.48632488)}'''

    dates = list(stock_df['date'])
    # get the last date
    predict_stock_date = dates[-1]

    dates = dates

    return dates, prices, prediction_models_outputs, predict_stock_date, test_stock_price,close_prices,l_close_price
