from unittest import result

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import row

import statsmodels.api as sm
import patsy
import mplcursors

data_demand = pd.read_csv('../demand_areas (1).csv')
Blindern_Temp_t4t = pd.read_csv('../Blindern_temperatur_t4t.csv')
data_price = pd.read_csv('../prices.csv')

def prisfolsomhet_time(data_demand, data_price, Blindern_Temp_t4t):
    start_date = '2021-09-01'
    end_date = '2022-03-31'

    data_demand_filtered = data_demand[(data_demand['Price_area'] == 'NO1') &
                                       (data_demand['Date'] >= start_date) &
                                       (data_demand['Date'] <= end_date)].copy()
    data_demand_filtered['Date'] = pd.to_datetime(data_demand_filtered['Date'])
    data_demand_filtered['Hour'] = data_demand_filtered['Hour'].astype(int)

    data_price_filtered = data_price[(data_price['Price_area'] == 'NO1') & (data_price['Date'] >= start_date) & (data_price['Date'] <= end_date)].copy()
    data_price_filtered['Date'] = pd.to_datetime(data_price_filtered['Date'])
    data_price_filtered['Hour'] = data_price_filtered['Hour'].astype(int)

    Blindern_Temp_t4t['Date'] = pd.to_datetime(Blindern_Temp_t4t['Date'])
    Blindern_Temp_t4t['Hour'] = Blindern_Temp_t4t['Hour'].astype(int)

    Blindern_Temp_t4t['Temperatur24'] = Blindern_Temp_t4t['Temperatur'].rolling(window=24, min_periods=1).mean()
    Blindern_Temp_t4t['Temperatur72'] = Blindern_Temp_t4t['Temperatur'].rolling(window=72, min_periods=1).mean()

    temp_filtered = Blindern_Temp_t4t[
        (Blindern_Temp_t4t['Date'] >= start_date) & (Blindern_Temp_t4t['Date'] <= end_date)]

    merge_1 = pd.merge(data_demand_filtered,data_price_filtered, on = ['Date', 'Hour', 'Price_area'])
    merge = pd.merge(merge_1, temp_filtered, on = ['Date', 'Hour'])

    '''pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_rows', None)
    print(merge)'''

    df = pd.DataFrame(merge)

    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Month'] = df['Date'].dt.strftime('%B')

    df['Hour'] = df['Hour'].astype(str)
    df['Hour'] = pd.Categorical(df['Hour'], categories=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                                                        '11', '12', '13', '14', '15', '16', '17', '18', '19',
                                                        '20', '21', '22', '23', '24'], ordered=True)

    df['Month'] = pd.Categorical(df['Month'], categories=['January', 'February', 'March', 'April', 'May', 'June',
                                                          'July', 'August', 'September', 'October', 'November',
                                                          'December'], ordered=True)

    y, X = patsy.dmatrices('np.log(Demand_kWh_avg) ~ Price_NOK_kWh + Temperatur24 + '
                           'I(Temperatur24**2) + I(Temperatur24**3) + Temperatur72 + '
                           'C(Hour, Treatment(reference="1")) + C(Month, Treatment(reference = "September")) + '
                           'C(Hour, Treatment(reference="1")) * Temperatur72',
                           data=df, return_type='dataframe', NA_action='drop')

    model = sm.OLS(y, X).fit()
    print(model.summary())

    # plott:
    ''' 
    beta_2 = model.params['Temperatur24']
    beta_3 = model.params['I(Temperatur24 ** 2)']
    beta_4 = model.params['I(Temperatur24 ** 3)']

    hour_1 = 0
    hour_2 = model.params['C(Hour, Treatment(reference="1"))[T.2]']
    hour_3 = model.params['C(Hour, Treatment(reference="1"))[T.3]']
    hour_4 = model.params['C(Hour, Treatment(reference="1"))[T.4]']
    hour_5 = model.params['C(Hour, Treatment(reference="1"))[T.5]']
    hour_6 = model.params['C(Hour, Treatment(reference="1"))[T.6]']
    hour_7 = model.params['C(Hour, Treatment(reference="1"))[T.7]']
    hour_8 = model.params['C(Hour, Treatment(reference="1"))[T.8]']
    hour_9 = model.params['C(Hour, Treatment(reference="1"))[T.9]']
    hour_10 = model.params['C(Hour, Treatment(reference="1"))[T.10]']
    hour_11 = model.params['C(Hour, Treatment(reference="1"))[T.11]']
    hour_12 = model.params['C(Hour, Treatment(reference="1"))[T.12]']
    hour_13 = model.params['C(Hour, Treatment(reference="1"))[T.13]']
    hour_14 = model.params['C(Hour, Treatment(reference="1"))[T.14]']
    hour_15 = model.params['C(Hour, Treatment(reference="1"))[T.15]']
    hour_16 = model.params['C(Hour, Treatment(reference="1"))[T.16]']
    hour_17 = model.params['C(Hour, Treatment(reference="1"))[T.17]']
    hour_16 = model.params['C(Hour, Treatment(reference="1"))[T.16]']
    hour_18 = model.params['C(Hour, Treatment(reference="1"))[T.18]']
    hour_19 = model.params['C(Hour, Treatment(reference="1"))[T.19]']
    hour_20 = model.params['C(Hour, Treatment(reference="1"))[T.20]']
    hour_21 = model.params['C(Hour, Treatment(reference="1"))[T.21]']
    hour_22 = model.params['C(Hour, Treatment(reference="1"))[T.22]']
    hour_23 = model.params['C(Hour, Treatment(reference="1"))[T.23]']
    hour_24 = model.params['C(Hour, Treatment(reference="1"))[T.24]']
    hour_list = [hour_1, float(hour_2), float(hour_3), float(hour_4), float(hour_5), float(hour_6), float(hour_7),
                 float(hour_8), float(hour_9), float(hour_10),
                 float(hour_11), float(hour_12), float(hour_13), float(hour_14), float(hour_15), float(hour_16),
                 float(hour_17), float(hour_18), float(hour_19), float(hour_20),
                 float(hour_21), float(hour_22), float(hour_23), float(hour_24)]

    hours = list(range(1, 25))
    temp_range = np.linspace(-20, 30, 200)
    temp_effect = beta_2 * temp_range + beta_3 * temp_range ** 2 + beta_4 * temp_range ** 3

    plt.figure(figsize=(10, 6))
    plt.plot(temp_range, temp_effect, color='green', linewidth=2)
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Temperatur (°C)')
    plt.ylabel('Kalkulert effekt (kWh)')
    plt.title('Effekt av Temperatur24 basert på beta-ene')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(hours, hour_list, color='green', linewidth=2)
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Timer')
    plt.ylabel('Beta-verdier for hver time')
    plt.title('Betaene til hver time mot timer i døgnet')
    plt.grid(True)
    plt.show()

    avg_hour_demand = data_demand_filtered.groupby('Hour')['Demand_kWh_avg'].mean()
    intercept = model.params['Intercept']

    hour_list_1 = [intercept]
    for h in range(2, 25):
        param_name = f'C(Hour, Treatment(reference= \"1\"))[T.{h}]'
        if param_name in model.params:
            hour_list_1.append(intercept + model.params[param_name])
        else:
            hour_list_1.append(intercept)

    hours_3 = list(range(1, 25))

    plt.figure(figsize=(10, 6))
    plt.plot(hours_3, avg_hour_demand, color='green', linewidth=2)
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Timer')
    plt.ylabel('Beta-verdier/ forbruk')
    plt.title('Betaene til hver time mot timer i døgnet')
    plt.grid(True)
    plt.legend()
    plt.show()'''


prisfolsomhet_time(data_demand, data_price, Blindern_Temp_t4t)