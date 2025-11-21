from unittest import result

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import row

import statsmodels.api as sm
import patsy
import mplcursors

data_demand = pd.read_csv('demand_areas (1).csv')
Blindern_Temp_t4t = pd.read_csv('Blindern_temperatur_t4t.csv')
data_price = pd.read_csv('prices.csv')

def prisfolsomhet_time(data_demand, data_price, Blindern_Temp_t4t):
    start_dato = '2021-09-01'
    end_dato = '2022-03-31'

    merge = pd.merge(data_demand,data_price, on = ['Date', 'Hour'])

    df = pd.DataFrame(merge)

    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_rows', None)
    print(df)


prisfolsomhet_time(data_demand, data_price, Blindern_Temp_t4t)