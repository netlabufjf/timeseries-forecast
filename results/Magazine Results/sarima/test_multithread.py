import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

def run_auto_feed_arima(args):
    to_train, to_test = args
    to_train_feed = to_train.copy()
    to_test = to_test.copy()
    predictions = []

    for _ in range(12):
        model = SARIMAX(to_train_feed.values, order=(2, 0, 3), seasonal_order=(2, 1, 1, 12))
        model = model.fit(disp=False)
        prediction = model.predict(start=len(to_train_feed), end=len(to_train_feed))
        predictions.append(prediction[0])

        # removing the first input from the test set
        to_test = to_test[1:]

        # append the last prediction to the train set
        to_train_feed = to_train_feed.append(pd.Series(prediction))

    predictions = np.array(predictions)

    return predictions

def run_auto_feed(train, test, predict_period=24, path='/'):
    to_train = train.copy()
    
    # montando as timeseries
    test_values = []
    train_values = []
    final_predictions = []
    for i in range(len(test)-predict_period):
        to_test = test[i:predict_period+i].copy()
        train_values.append(to_train)
        test_values.append(to_test.values)

        to_train = to_train.append(to_test)

    # params_list = [params]*len(train_values)

    p = Pool(cpu_count())
    predictions = list(tqdm(
        p.imap(run_auto_feed_arima, list(zip(train_values, test_values))),
        total=len(train_values)
    ))
    p.close()
    p.join()

    test_values = np.array(test_values)
    predictions = np.array(predictions)
    results = (test_values, predictions)
    pickle.dump(results, open( path+"auto_feed_results.p", "wb" ))


if __name__ == "__main__":
    # reading data
    c2g_data = pd.read_csv('../../data/demand_datasets/c2g_demand.csv', index_col=0)
    # cleaning process
    c2g_data.index = pd.to_datetime(c2g_data.index)
    
    init_period = '2016-12-13 15:00'
    end_period = '2017-02-25 17:00'

    c2g_data = c2g_data[(c2g_data.index >= init_period) & (c2g_data.index <= end_period)]
    
    c2g_train = c2g_data[c2g_data.index < '2017-02-11 03:00:00'].copy()
    c2g_test = c2g_data[c2g_data.index >= '2017-02-11 03:00:00'].copy()

    # use only the travels
    c2g_travels_train, c2g_travels_test = c2g_train.travels, c2g_test.travels

    sarima_params = {
        'order': (2, 0, 3), 
        'seasonal_order': (2, 1, 1, 12)
    }

    run_auto_feed(c2g_travels_train, c2g_travels_test, path="C:/Users/victo/Documents/PythonScripts/timeseries/JISA/sarima/")