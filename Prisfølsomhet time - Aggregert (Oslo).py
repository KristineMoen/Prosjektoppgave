from unittest import result

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import row

import statsmodels.api as sm
import patsy
import mplcursors

# -------------------------------------- LESER DATA --------------------------------------#

data_demand = pd.read_csv('/Users/kristinemoen/Documents/5-klasse/Prosjektoppgave_CSV_filer/demand.csv')

data_price = pd.read_csv('prices.csv')
data_price_update = data_price.drop(columns = ['Price_NOK_MWh'])

Blindern_Temp_t4t = pd.read_csv('Blindern_temperatur_t4t.csv')

#------------------------------------- FINNE AKTUELLE HUSSTANDER -------------------------------------------#

#Finne ID:
data_answer = pd.read_csv('answers.csv')
data_households = pd.read_csv('households (1).csv')
liste_husstander = []

def finne_husstander():
    for index, rad in data_answer.iterrows():
        if (
                rad["Q_City"] in [4, 1, 2]   and    # 4 = Oslo, 2 = Lillestrøm, 1 = Bærum
                #rad["Q22"] == 1            # 1 = Enebolig 4 = Boligblokk
                #rad["Q23"] == 9         # 1= Under 30 kvm, 2 = 30-49 kvm, 3 = 50-59 kvm, 4 = 60-79 kvm, 5 = 80-99 kvm, 6 = 100-119 kvm, 7 = 120-159 kvm, 8 = 160-199 kvm, 9 = 200 kvm eller større, 10 = vet ikke
                #rad["Q21"] == 6         # 1 = Under 300 000 kr, 2 = 300 000 - 499 999, 3 = 500 000 -799 999, 4 = 800 000 - 999 999, 5 = 1 000 000 - 1 499 999, 6 = 1 500 000 eller mer, 7 = Vil ikke oppgi, 8 = Vet ikke
                #rad["Q20"] == 4         # 1 = Ingen fullført utdanning, 2 = Grunnskole, 3 = Vgs, 4 = Høyskole/Uni lavere grad, 5 = Høyskol/Uni høyere grad
                #rad["Q1"] == 1          # 1 = Fulgte med på egen strømbruk, 2 = følgte ikke med
                #rad['Q4'] == 4
                #rad["Q29"] == 1        # 1 = Ja, 2 = Nei
                #rad["Q8_12"] == 0
                #rad["Q7"] == 3
                rad["Q29"] == 2
        ):

            # Sjekk om ID finnes i data_households og har Demand_data = 'Yes'
            id_verdi = rad["ID"]
            match = data_households[
                (data_households["ID"] == id_verdi) &
                (data_households["Demand_data"] == "Yes")
                ]
            if not match.empty:
                liste_husstander.append(int(id_verdi))

    print("ID-er som oppfyller kravene:", len(liste_husstander))

finne_husstander()

#--------------------------------- REGNE PÅ PRISFØLSOMHET PER TIME FOR TIME ------------------------------------------#

#test_liste_husstander = [512, 642] #Bare for test

#-----------------------------------------------------------------------------------

'''Regresjon for "direkte", ren regresjonsanalyse: demand = beta_0 + beta_1 *pris + beta_2 * Temperatur24 + beta_3 * Temepartur24^2 + Beta_4 * Temperatur24^3
                                                             + Temperatur72 + Hour_i + Month + Hour_i * Temperatur72 + error'''

def direkte_prisfolsomhet_time(liste_husstander, data_demand, data_price_update, data_households, Blindern_Temp_t4t):
    start_dato = '2021-09-01'
    end_dato ='2022-03-31'

    # Gjennomsnits demand per dag for alle ID-ene:
    data_demand['Date'] = pd.to_datetime(data_demand['Date'])
    data_demand['Hour'] = data_demand['Hour'].astype(int)
    demand_data_filtered = data_demand[(data_demand['ID'].isin(liste_husstander)) &
                                       (data_demand['Date'] >= start_dato) &
                                       (data_demand['Date'] <= end_dato)].copy()

    total_hour_demand = demand_data_filtered.groupby(['Date','Hour'])['Demand_kWh'].sum().reset_index()


    # Gjennosnits prisen:
    price_area = data_households[data_households['ID'].isin(liste_husstander)].iloc[0]['Price_area']
    price_data = data_price_update[data_price_update['Price_area'] == price_area].copy()
    price_data['Date'] = pd.to_datetime(price_data['Date'])
    price_data['Hour'] = price_data['Hour'].astype(int)
    price_filtered = price_data[(price_data['Date'] >= start_dato) & (price_data['Date'] <= end_dato)]
    price_filtered.loc[:, 'Price_NOK_kWh'] = price_filtered['Price_NOK_kWh'].apply(
        lambda x: x if x > 0 else 0.01)  # Dette skal fikset prisen, om den er negativ

    # Gjennomsnits temoperatur:
    Blindern_Temp_t4t['Date'] = pd.to_datetime(Blindern_Temp_t4t['Date'])
    Blindern_Temp_t4t['Hour'] = Blindern_Temp_t4t['Hour'].astype(int)

    Blindern_Temp_t4t['Temperatur24'] = Blindern_Temp_t4t['Temperatur'].rolling(window=24, min_periods=1).mean()
    Blindern_Temp_t4t['Temperatur72'] = Blindern_Temp_t4t['Temperatur'].rolling(window=72, min_periods=1).mean()

    temp_filtered = Blindern_Temp_t4t[
        (Blindern_Temp_t4t['Date'] >= start_dato) & (Blindern_Temp_t4t['Date'] <= end_dato)]

    # Merge:
    merged_1 = pd.merge(total_hour_demand, price_filtered, on=['Date', 'Hour'])
    merged = pd.merge(merged_1, temp_filtered, on=['Date', 'Hour'])

    filtered = merged[(merged['Demand_kWh'] > 0) & (merged['Price_NOK_kWh'] > 0) & (
        merged['Temperatur'].notnull())].copy()

    df = pd.DataFrame(filtered)

    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Month'] = df['Date'].dt.strftime('%B')

    df['Wday'] = df['Date'].dt.weekday.astype(str)

    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_rows', None)
    #print(df)

    # Beregninger:
    df['Hour'] = df['Hour'].astype(str)
    df['Hour'] = pd.Categorical(df['Hour'], categories=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                                                        '11', '12', '13', '14', '15', '16', '17', '18', '19',
                                                        '20', '21', '22', '23', '24'], ordered=True)

    df['Month'] = pd.Categorical(df['Month'], categories=['January', 'February', 'March', 'April', 'May', 'June',
                                                          'July', 'August', 'September', 'October', 'November',
                                                          'December'], ordered=True)

    # 0 = mandag, 1 = tirsdag, 2 = onsdag, 3 = torsdag, 4 = fredag, 5 = lørdag og 6 = søndag
    df['Wday'] = pd.Categorical(df['Wday'], categories=['0', '1', '2', '3', '4', '5', '6'], ordered=True)

    y, X = patsy.dmatrices('Demand_kWh ~ Price_NOK_kWh + Temperatur24 + '
                           'I(Temperatur24**2) + I(Temperatur24**3) + Temperatur72 + '
                           'C(Hour, Treatment(reference="1")) + C(Month, Treatment(reference = "September")) + '
                           'C(Wday, Treatment(reference = "0")) + '
                           'C(Hour, Treatment(reference="1")) * Temperatur72 + '
                           'C(Wday, Treatment(reference = "0")) * C(Hour, Treatment(reference="1")) + '
                           'C(Month, Treatment(reference = "September")) * C(Wday, Treatment(reference = "0"))',
                           data=df, return_type='dataframe', NA_action='drop')



    model = sm.OLS(y, X).fit()
    print(model.summary())

    # plott:
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
    hour_list = [hour_1, float(hour_2), float(hour_3), float(hour_4), float(hour_5), float(hour_6), float(hour_7), float(hour_8), float(hour_9), float(hour_10),
                      float(hour_11), float(hour_12), float(hour_13), float(hour_14), float(hour_15), float(hour_16), float(hour_17), float(hour_18), float(hour_19), float(hour_20),
                      float(hour_21), float(hour_22), float(hour_23), float(hour_24)]

    hours = list(range(1,25))
    temp_range = np.linspace(-20, 30, 200)
    temp_effect = beta_2 *temp_range + beta_3 *temp_range**2 + beta_4 * temp_range**3

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

    avg_hour_demand = total_hour_demand.groupby('Hour')['Demand_kWh'].mean()
    intercept = model.params['Intercept']

    hour_list_1 = [intercept]
    for h in range(2,25):
        param_name = f'C(Hour, Treatment(reference= \"1\"))[T.{h}]'
        if param_name in model.params:
            hour_list_1.append(intercept + model.params[param_name])
        else:
            hour_list_1.append(intercept)

    hours_3 = list(range(1,25))

    plt.figure(figsize=(10, 6))
    plt.plot(hours_3, avg_hour_demand, color='green', linewidth=2)
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Timer')
    plt.ylabel('Beta-verdier/ forbruk')
    plt.title('Betaene til hver time mot timer i døgnet')
    plt.grid(True)
    plt.legend()
    plt.show()





# -----------------------------------------------------------------------------------
'''Regresjon for log-lin/ lin-log: 

      1) demand = beta_0 + beta_1 *log(pris) + beta_2 *Temperatur24 + beta_3 *Temepartur24^2 + Beta_4 *Temperatur24^3 + 
                        + Temperatur72 + Hour_i + Month + Hour_i * Temperatur72 + error


      2) log(demand) = beta_0 + beta_1 * pris + beta_2 *Temperatur24 + beta_3 *Temepartur24^2 + Beta_4 *Temperatur24^3 + 
                        + Temperatur72 + Hour_i + Month + Hour_i * Temperatur72 + error
'''

def lin_log_prisfolsomhet_t4t(liste_husstander, data_demand, data_price_update, data_households, Blindern_Temp_t4t):
    start_dato = '2021-08-01'
    end_dato = '2021-12-31'
    start_dato = pd.to_datetime(start_dato)
    end_dato = pd.to_datetime(end_dato)

    # Gjennomsnits demand per dag for alle ID-ene:
    data_demand['Date'] = pd.to_datetime(data_demand['Date'])
    data_demand['Hour'] = data_demand['Hour'].astype(int)
    demand_data_filtered = data_demand[(data_demand['ID'].isin(liste_husstander)) &
                                       (data_demand['Date'] >= start_dato) &
                                       (data_demand['Date'] <= end_dato)].copy()

    total_hour_demand = demand_data_filtered.groupby(['Date', 'Hour'])['Demand_kWh'].sum().reset_index()

    # Gjennosnits prisen:
    price_area = data_households[data_households['ID'].isin(liste_husstander)].iloc[0]['Price_area']
    price_data = data_price_update[data_price_update['Price_area'] == price_area].copy()
    price_data['Date'] = pd.to_datetime(price_data['Date'])
    price_data['Hour'] = price_data['Hour'].astype(int)
    price_filtered = price_data[(price_data['Date'] >= start_dato) & (price_data['Date'] <= end_dato)]
    price_filtered.loc[:, 'Price_NOK_kWh'] = price_filtered['Price_NOK_kWh'].apply(
        lambda x: x if x > 0 else 0.01)  # Dette skal fikset prisen, om den er negativ

    # Gjennomsnits temoperatur:
    Blindern_Temp_t4t['Date'] = pd.to_datetime(Blindern_Temp_t4t['Date'])
    Blindern_Temp_t4t['Hour'] = Blindern_Temp_t4t['Hour'].astype(int)

    Blindern_Temp_t4t['Temperatur24'] = Blindern_Temp_t4t['Temperatur'].rolling(window=24, min_periods=1).mean()
    Blindern_Temp_t4t['Temperatur72'] = Blindern_Temp_t4t['Temperatur'].rolling(window=72, min_periods=1).mean()

    temp_filtered = Blindern_Temp_t4t[
        (Blindern_Temp_t4t['Date'] >= start_dato) & (Blindern_Temp_t4t['Date'] <= end_dato)]

    # Merge:
    merged_1 = pd.merge(total_hour_demand, price_filtered, on=['Date', 'Hour'])
    merged = pd.merge(merged_1, temp_filtered, on=['Date', 'Hour'])

    filtered = merged[(merged['Demand_kWh'] > 0) & (merged['Price_NOK_kWh'] > 0) & (
        merged['Temperatur'].notnull())].copy()

    df = pd.DataFrame(filtered)

    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Month'] = df['Date'].dt.strftime('%B')

    df['Wday'] = df['Date'].dt.weekday.astype(str)

    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_rows', None)
    # print(df)

    # Beregninger:
    df['Hour'] = df['Hour'].astype(str)
    df['Hour'] = pd.Categorical(df['Hour'], categories=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                                                        '11', '12', '13', '14', '15', '16', '17', '18', '19',
                                                        '20', '21', '22', '23', '24'], ordered=True)

    df['Month'] = pd.Categorical(df['Month'], categories=['January', 'February', 'March', 'April', 'May', 'June',
                                                          'July', 'August', 'September', 'October', 'November',
                                                          'December'], ordered=True)

    # 0 = mandag, 1 = tirsdag, 2 = onsdag, 3 = torsdag, 4 = fredag, 5 = lørdag og 6 = søndag
    df['Wday'] = pd.Categorical(df['Wday'], categories=['0', '1', '2', '3', '4', '5', '6'], ordered=True)

    y, X = patsy.dmatrices('Demand_kWh ~ np.log(Price_NOK_kWh) + Temperatur24 + '
                            'I(Temperatur24**2) + I(Temperatur24**3) + Temperatur72 + '
                            'C(Wday, Treatment(reference = "0")) + '
                            'C(Hour, Treatment(reference="1")) + C(Month, Treatment(reference = "April")) +'
                            'C(Hour, Treatment(reference="1")) * Temperatur72 + '
                           'C(Wday, Treatment(reference = "0")) * C(Hour, Treatment(reference="1"))',
                           data=df, return_type='dataframe', NA_action='drop')

    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 24})
    print(model.summary())


def log_lin_prisfolsomhet_t4t(liste_husstander,data_demand,data_price_update,data_households, Blindern_Temp_t4t):
    start_dato = '2021-09-01'
    end_dato = '2021-12-31'
    start_dato = pd.to_datetime(start_dato)
    end_dato = pd.to_datetime(end_dato)

    # Gjennomsnits demand per dag for alle ID-ene:
    data_demand['Date'] = pd.to_datetime(data_demand['Date'])
    data_demand['Hour'] = data_demand['Hour'].astype(int)
    demand_data_filtered = data_demand[(data_demand['ID'].isin(liste_husstander)) &
                                       (data_demand['Date'] >= start_dato) &
                                       (data_demand['Date'] <= end_dato)].copy()

    total_hour_demand = demand_data_filtered.groupby(['Date', 'Hour'])['Demand_kWh'].sum().reset_index()

    # Gjennosnits prisen:
    price_area = data_households[data_households['ID'].isin(liste_husstander)].iloc[0]['Price_area']
    price_data = data_price_update[data_price_update['Price_area'] == price_area].copy()
    price_data['Date'] = pd.to_datetime(price_data['Date'])
    price_data['Hour'] = price_data['Hour'].astype(int)
    price_filtered = price_data[(price_data['Date'] >= start_dato) & (price_data['Date'] <= end_dato)]
    price_filtered.loc[:, 'Price_NOK_kWh'] = price_filtered['Price_NOK_kWh'].apply(
        lambda x: x if x > 0 else 0.01)  # Dette skal fikset prisen, om den er negativ

    # Gjennomsnits temoperatur:
    Blindern_Temp_t4t['Date'] = pd.to_datetime(Blindern_Temp_t4t['Date'])
    Blindern_Temp_t4t['Hour'] = Blindern_Temp_t4t['Hour'].astype(int)

    Blindern_Temp_t4t['Temperatur24'] = Blindern_Temp_t4t['Temperatur'].rolling(window=24, min_periods=1).mean()
    Blindern_Temp_t4t['Temperatur72'] = Blindern_Temp_t4t['Temperatur'].rolling(window=72, min_periods=1).mean()

    temp_filtered = Blindern_Temp_t4t[
        (Blindern_Temp_t4t['Date'] >= start_dato) & (Blindern_Temp_t4t['Date'] <= end_dato)]

    # Merge:
    merged_1 = pd.merge(total_hour_demand, price_filtered, on=['Date', 'Hour'])
    merged = pd.merge(merged_1, temp_filtered, on=['Date', 'Hour'])

    filtered = merged[(merged['Demand_kWh'] > 0) & (merged['Price_NOK_kWh'] > 0) & (
        merged['Temperatur'].notnull())].copy()

    df = pd.DataFrame(filtered)

    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Month'] = df['Date'].dt.strftime('%B')

    df['Wday'] = df['Date'].dt.weekday.astype(str)

    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_rows', None)
    # print(df)

    # Beregninger:
    df['Hour'] = df['Hour'].astype(str)
    df['Hour'] = pd.Categorical(df['Hour'], categories=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                                                        '11', '12', '13', '14', '15', '16', '17', '18', '19',
                                                        '20', '21', '22', '23', '24'], ordered=True)

    df['Month'] = pd.Categorical(df['Month'], categories=['January', 'February', 'March', 'April', 'May', 'June',
                                                          'July', 'August', 'September', 'October', 'November',
                                                          'December'], ordered=True)

    # 0 = mandag, 1 = tirsdag, 2 = onsdag, 3 = torsdag, 4 = fredag, 5 = lørdag og 6 = søndag
    df['Wday'] = pd.Categorical(df['Wday'], categories=['0', '1', '2', '3', '4', '5', '6'], ordered=True)

    y, X = patsy.dmatrices('np.log(Demand_kWh) ~ Price_NOK_kWh + Temperatur24 + '
                               'I(Temperatur24**2) + I(Temperatur24**3) + Temperatur72 + '
                               'C(Wday, Treatment(reference = "0")) + '
                            'C(Hour, Treatment(reference="1")) + C(Month, Treatment(reference = "April")) +'
                            'C(Hour, Treatment(reference="1")) * Temperatur72 + '
                           'C(Wday, Treatment(reference = "0")) * C(Hour, Treatment(reference="1"))',
                           data=df, return_type='dataframe', NA_action='drop')

    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 24})
    print(model.summary())

    # plott:
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

    avg_hour_demand = total_hour_demand.groupby('Hour')['Demand_kWh'].mean()
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
    plt.show()

#-----------------------------------------------------------------------------------

'''Regresjon for log-log: log(demand) = beta_0 + beta_1 *log(pris) + beta_2 *Temperatur24 + beta_3 *Temperatur24^2 + beta_4 *Temperatur24^3 
                                        + Temperatur72 + Hour_i + Month + Hour_i*Temperatur72 + error'''

def log_log_prisfolsomhet_t4t(liste_husstander, data_demand, data_price_update, data_households, Blindern_Temp_t4t):
    start_dato = '2021-10-01'
    end_dato = '2022-03-31'
    start_dato = pd.to_datetime(start_dato)
    end_dato = pd.to_datetime(end_dato)

    # Gjennomsnits demand per dag for alle ID-ene:
    data_demand['Date'] = pd.to_datetime(data_demand['Date'])
    data_demand['Hour'] = data_demand['Hour'].astype(int)
    demand_data_filtered = data_demand[(data_demand['ID'].isin(liste_husstander)) &
                                       (data_demand['Date'] >= start_dato) &
                                       (data_demand['Date'] <= end_dato)].copy()

    total_hour_demand = demand_data_filtered.groupby(['Date', 'Hour'])['Demand_kWh'].sum().reset_index()

    # Gjennosnits prisen:
    price_area = data_households[data_households['ID'].isin(liste_husstander)].iloc[0]['Price_area']
    price_data = data_price_update[data_price_update['Price_area'] == price_area].copy()
    price_data['Date'] = pd.to_datetime(price_data['Date'])
    price_data['Hour'] = price_data['Hour'].astype(int)
    price_filtered = price_data[(price_data['Date'] >= start_dato) & (price_data['Date'] <= end_dato)]
    price_filtered.loc[:, 'Price_NOK_kWh'] = price_filtered['Price_NOK_kWh'].apply(
        lambda x: x if x > 0 else 0.01)  # Dette skal fikset prisen, om den er negativ

    # Gjennomsnits temoperatur:
    Blindern_Temp_t4t['Date'] = pd.to_datetime(Blindern_Temp_t4t['Date'])
    Blindern_Temp_t4t['Hour'] = Blindern_Temp_t4t['Hour'].astype(int)

    Blindern_Temp_t4t['Temperatur24'] = Blindern_Temp_t4t['Temperatur'].rolling(window=24, min_periods=1).mean()
    Blindern_Temp_t4t['Temperatur72'] = Blindern_Temp_t4t['Temperatur'].rolling(window=72, min_periods=1).mean()

    temp_filtered = Blindern_Temp_t4t[
        (Blindern_Temp_t4t['Date'] >= start_dato) & (Blindern_Temp_t4t['Date'] <= end_dato)]

    # Merge:
    merged_1 = pd.merge(total_hour_demand, price_filtered, on=['Date', 'Hour'])
    merged = pd.merge(merged_1, temp_filtered, on=['Date', 'Hour'])

    filtered = merged[(merged['Demand_kWh'] > 0) & (merged['Price_NOK_kWh'] > 0) & (
        merged['Temperatur'].notnull())].copy()

    df = pd.DataFrame(filtered)

    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Month'] = df['Date'].dt.strftime('%B')

    df['Wday'] = df['Date'].dt.weekday.astype(str)

    #pd.set_option('display.max_colwidth', None)
    #pd.set_option('display.width', None)
    #pd.set_option('display.max_rows', None)
    #print(df)

    # Beregninger:
    df['Hour'] = df['Hour'].astype(str)
    df['Hour'] = pd.Categorical(df['Hour'], categories=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                                                        '11', '12', '13', '14', '15', '16', '17', '18', '19',
                                                        '20', '21', '22', '23', '24'], ordered=True)

    df['Month'] = pd.Categorical(df['Month'], categories=['January', 'February', 'March', 'April', 'May', 'June',
                                                          'July', 'August', 'September', 'October', 'November',
                                                          'December'], ordered=True)

    # 0 = mandag, 1 = tirsdag, 2 = onsdag, 3 = torsdag, 4 = fredag, 5 = lørdag og 6 = søndag
    df['Wday'] = pd.Categorical(df['Wday'], categories=['0', '1', '2', '3', '4', '5', '6'], ordered=True)

    y, X = patsy.dmatrices('np.log(Demand_kWh) ~ np.log(Price_NOK_kWh) + Temperatur24 + '
                           'I(Temperatur24**2) + I(Temperatur24**3) + Temperatur72 + '
                           'C(Wday, Treatment(reference = "0")) + '
                            'C(Hour, Treatment(reference="1")) + C(Month, Treatment(reference = "October")) +'
                            'C(Hour, Treatment(reference="1")) * Temperatur72 + '
                           'C(Wday, Treatment(reference = "0")) * C(Hour, Treatment(reference="1")) + '
                           'C(Month, Treatment(reference = "October")) * C(Wday, Treatment(reference = "0"))',
                           data=df, return_type='dataframe', NA_action='drop')

    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 24})
    print(model.summary())


#-----------------------------------------------------------------------------------

'''Kjøre funksjonene, printer ut resultatene '''

resultater = direkte_prisfolsomhet_time(liste_husstander,data_demand,data_price_update,data_households,Blindern_Temp_t4t)
#resultater = lin_log_prisfolsomhet_t4t(liste_husstander, data_demand, data_price_update, data_households, Blindern_Temp_t4t)
#resultater = log_lin_prisfolsomhet_t4t(liste_husstander, data_demand, data_price_update, data_households, Blindern_Temp_t4t)
#resultater = log_log_prisfolsomhet_t4t(liste_husstander, data_demand, data_price_update, data_households, Blindern_Temp_t4t)

