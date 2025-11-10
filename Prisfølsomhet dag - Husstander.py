import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import row

import statsmodels.api as sm
import mplcursors
import patsy

# -------------------------------------- LESER DATA --------------------------------------#

data_demand = pd.read_csv('/Users/kristinemoen/Documents/5-klasse/Prosjektoppgave_CSV_filer/demand.csv')

data_price = pd.read_csv('prices.csv')
data_price_update = data_price.drop(columns = ['Price_NOK_MWh'])

Blindern_Temperatur_dag = pd.read_csv('Blindern_Temperatur_dag.csv')

#------------------------------------- FINNE AKTUELLE HUSSTANDER -------------------------------------------#

#Finne ID:
data_answer = pd.read_csv('answers.csv')
data_households = pd.read_csv('households (1).csv')
liste_husstander = []

def finne_husstander():
    for index, rad in data_answer.iterrows():
        if (
                rad["Q_City"] == 4 and      # 4 = Oslo 5 = Bergen 6 = Tromsø 7 = Trondheim
                rad["Q22"] == 1 and        # 1 = Enebolig 4 = Boligblokk
                rad["Q23"] == 9 and        # 1= Under 30 kvm, 2 = 30-49 kvm, 3 = 50-59 kvm, 4 = 60-79 kvm, 5 = 80-99 kvm, 6 = 100-119 kvm, 7 = 120-159 kvm, 8 = 160-199 kvm, 9 = 200 kvm eller større, 10 = vet ikke
                rad["Q21"] == 6         # 1 = Under 300 000 kr, 2 = 300 000 - 499 999, 3 = 500 000 -799 999, 4 = 800 000 - 999 999, 5 = 1 000 000 - 1 499 999, 6 = 1 500 000 eller mer, 7 = Vil ikke oppgi, 8 = Vet ikke
                #rad["Q20"] == 4         # 1 = Ingen fullført utdanning, 2 = Grunnskole, 3 = Vgs, 4 = Høyskole/Uni lavere grad, 5 = Høyskol/Uni høyere grad
                #rad["Q20"] == 5
        ):

            # Sjekk om ID finnes i data_households og har Demand_data = 'Yes'
            id_verdi = rad["ID"]
            match = data_households[
                (data_households["ID"] == id_verdi) &
                (data_households["Demand_data"] == "Yes")
                ]
            if not match.empty:
                liste_husstander.append(int(id_verdi))

    print("ID-er som oppfyller kravene:", liste_husstander)

finne_husstander()

#--------------------------------- REGNE PÅ PRISFØLSOMHET PER DAG ------------------------------------------#

test_liste_husstander = [512] #Bare for test

#-----------------------------------------------------------------------------------

'''Regresjon for log-log: log(demand) = beta_0 + beta_1 *log(pris) + beta_2 *T + beta_3 *T^2 + beta_4 *T^3 + Temperatur3 + 
                                        Month + Month*Temperatur3'''

def log_log_prisfølsomhet_dag(test_liste_husstander, data_demand, data_price_update, data_households, Blindern_Temperatur_dag):
    start_dato = '2021-04-01'
    end_dato = '2022-03-31'

    for ID in test_liste_husstander:
        #Gjennomsnits demand per dag:
        demand_ID = data_demand[data_demand['ID'] == ID].copy()
        demand_ID['Date'] = pd.to_datetime(demand_ID['Date'])
        demand_filtered = demand_ID[(demand_ID['Date'] >= start_dato) & (demand_ID['Date'] <= end_dato)]

        avg_demand_per_day = demand_filtered.groupby('Date')['Demand_kWh'].sum().reset_index()
        avg_demand_per_day['Avg demand kWh per day'] = avg_demand_per_day['Demand_kWh']/24

        #Gjennomsnitss pris per dag:
        price_area = data_households[data_households['ID'] == ID].iloc[0]['Price_area']
        price_data = data_price_update[data_price_update['Price_area'] == price_area].copy()
        price_data['Date'] = pd.to_datetime(price_data['Date'])
        price_filtered = price_data[(price_data['Date'] >= start_dato) & (price_data['Date'] <= end_dato)]

        avg_price_per_day = price_filtered.groupby('Date')['Price_NOK_kWh'].sum().reset_index()
        avg_price_per_day['Avg price NOK kWh per day'] = avg_price_per_day['Price_NOK_kWh']/24

        #Temperatur:
        Blindern_Temperatur_dag['Date'] = pd.to_datetime(Blindern_Temperatur_dag['Date'])

        Blindern_Temperatur_dag['Temperatur3'] = Blindern_Temperatur_dag['Temperatur'].rolling(window=3, min_periods=1).mean()

        temp_filtered = Blindern_Temperatur_dag[(Blindern_Temperatur_dag['Date'] >= start_dato) & (Blindern_Temperatur_dag['Date'] <= end_dato)]

        #Merge datasettene til et stort et:
        merged_1 = pd.merge(avg_demand_per_day, avg_price_per_day, on = 'Date')
        merged_1['ID'] = ID
        merged = pd.merge(merged_1, temp_filtered, on = 'Date')


        filtered = merged[(merged['Avg demand kWh per day'] > 0) & (merged['Avg price NOK kWh per day'] > 0) & (merged['Temperatur'].notnull())].copy()

        df = pd.DataFrame(filtered)
        cols = ['ID'] + [col for col in df.columns if col != 'ID']
        df = df[cols]

        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        df['Month'] = df['Date'].dt.strftime('%B')

        #pd.set_option('display.max_colwidth', None)
        #pd.set_option('display.width', None)
        #pd.set_option('display.max_rows', None)
        #print(df)

        # Kjører regresjonsanalysen:

        df['Month'] = pd.Categorical(df['Month'], categories=['January', 'February', 'March', 'April', 'May', 'June',
                                                              'July', 'August', 'September', 'October', 'November',
                                                              'December'], ordered=True)

        y, X = patsy.dmatrices('np.log(Q("Avg demand kWh per day")) ~ np.log(Q("Avg price NOK kWh per day")) + Temperatur + '
                               'I(Temperatur**2) + I(Temperatur**3) + Temperatur3 + '
                               'C(Month, Treatment(reference = "April")) + C(Month, Treatment(reference = "April")) * Temperatur3',
                               data=df, return_type='dataframe', NA_action='drop')

        model = sm.OLS(y, X).fit()
        print(model.summary())

#-----------------------------------------------------------------------------------

'''Regresjon for log-lin/ lin-log: 
      1) log(demand) = beta_0 + beta_1 *T + beta_2 *T^2 + beta_3 *T^3 + beta_4 *pris +
                        Temperatur3 + Month + Month*Temperatur3
                        
      2) demand = beta_0 + beta_1 *log(pris) + beta_2 *T + beta_3 *T^2 + Beta_4 *T^3 +
                    Temperatur3 + Month + Month*Temperatur3
      
      3) log(demand) = beta_0 + beta_1 *pris + beta_2 *T + beta_3 *T^2 + Beta_4 *T^3 +
                    Temperatur3 + Month + Month*Temperatur3
'''

def log_lin_tempfølsomhet_temp_dag(test_liste_husstander,data_demand, data_price_update, data_households, Blindern_Temperatur_dag):
    start_dato = '2021-04-01'
    end_dato = '2022-03-31'

    for ID in test_liste_husstander:
        # Gjennomsnits demand per dag:
        demand_ID = data_demand[data_demand['ID'] == ID].copy()
        demand_ID['Date'] = pd.to_datetime(demand_ID['Date'])
        demand_filtered = demand_ID[(demand_ID['Date'] >= start_dato) & (demand_ID['Date'] <= end_dato)]

        avg_demand_per_day = demand_filtered.groupby('Date')['Demand_kWh'].sum().reset_index()
        avg_demand_per_day['Avg demand kWh per day'] = avg_demand_per_day['Demand_kWh'] / 24

        # Gjennomsnitss pris per dag:
        price_area = data_households[data_households['ID'] == ID].iloc[0]['Price_area']
        price_data = data_price_update[data_price_update['Price_area'] == price_area].copy()
        price_data['Date'] = pd.to_datetime(price_data['Date'])
        price_filtered = price_data[(price_data['Date'] >= start_dato) & (price_data['Date'] <= end_dato)]

        avg_price_per_day = price_filtered.groupby('Date')['Price_NOK_kWh'].sum().reset_index()
        avg_price_per_day['Avg price NOK kWh per day'] = avg_price_per_day['Price_NOK_kWh'] / 24

        # Temperatur:
        Blindern_Temperatur_dag['Date'] = pd.to_datetime(Blindern_Temperatur_dag['Date'])

        Blindern_Temperatur_dag['Temperatur3'] = Blindern_Temperatur_dag['Temperatur'].rolling(window=3, min_periods=1).mean()
        temp_filtered = Blindern_Temperatur_dag[
            (Blindern_Temperatur_dag['Date'] >= start_dato) & (Blindern_Temperatur_dag['Date'] <= end_dato)]

        # Merge datasettene til et stort et:
        merged_1 = pd.merge(avg_demand_per_day, avg_price_per_day, on='Date')
        merged_1['ID'] = ID
        merged = pd.merge(merged_1, temp_filtered, on='Date')

        filtered = merged[(merged['Avg demand kWh per day'] > 0) & (merged['Avg price NOK kWh per day'] > 0) & (
            merged['Temperatur'].notnull())].copy()

        df = pd.DataFrame(filtered)
        cols = ['ID'] + [col for col in df.columns if col != 'ID']
        df = df[cols]

        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        df['Month'] = df['Date'].dt.strftime('%B')

        # pd.set_option('display.max_colwidth', None)
        # pd.set_option('display.width', None)
        # pd.set_option('display.max_rows', None)
        #print(df)

        # Kjører regresjonsanalysen:

        df['Month'] = pd.Categorical(df['Month'], categories=['January', 'February', 'March', 'April', 'May', 'June',
                                                              'July', 'August', 'September', 'October', 'November',
                                                              'December'], ordered=True)

        y, X = patsy.dmatrices('np.log(Q("Avg demand kWh per day")) ~ Temperatur + '
                               'I(Temperatur**2) + I(Temperatur**3) + Temperatur3 +  '
                               'C(Month, Treatment(reference = "April")) + C(Month, Treatment(reference = "April")) * Temperatur3',
                               data=df, return_type='dataframe', NA_action='drop')

        model = sm.OLS(y, X).fit()
        print(model.summary())

def lin_log_prisfølsomhet_pris_dag(test_liste_husstander, data_demand, data_price_update, data_households, Blindern_Temperatur_dag):
    start_dato = '2021-04-01'
    end_dato = '2022-03-31'

    for ID in test_liste_husstander:
        # Gjennomsnits demand per dag:
        demand_ID = data_demand[data_demand['ID'] == ID].copy()
        demand_ID['Date'] = pd.to_datetime(demand_ID['Date'])
        demand_filtered = demand_ID[(demand_ID['Date'] >= start_dato) & (demand_ID['Date'] <= end_dato)]

        avg_demand_per_day = demand_filtered.groupby('Date')['Demand_kWh'].sum().reset_index()
        avg_demand_per_day['Avg demand kWh per day'] = avg_demand_per_day['Demand_kWh'] / 24

        # Gjennomsnitss pris per dag:
        price_area = data_households[data_households['ID'] == ID].iloc[0]['Price_area']
        price_data = data_price_update[data_price_update['Price_area'] == price_area].copy()
        price_data['Date'] = pd.to_datetime(price_data['Date'])
        price_filtered = price_data[(price_data['Date'] >= start_dato) & (price_data['Date'] <= end_dato)]

        avg_price_per_day = price_filtered.groupby('Date')['Price_NOK_kWh'].sum().reset_index()
        avg_price_per_day['Avg price NOK kWh per day'] = avg_price_per_day['Price_NOK_kWh'] / 24

        # Temperatur:
        Blindern_Temperatur_dag['Date'] = pd.to_datetime(Blindern_Temperatur_dag['Date'])
        Blindern_Temperatur_dag['Temperatur3'] = Blindern_Temperatur_dag['Temperatur'].rolling(window=3,
                                                                                               min_periods=1).mean()
        temp_filtered = Blindern_Temperatur_dag[
            (Blindern_Temperatur_dag['Date'] >= start_dato) & (Blindern_Temperatur_dag['Date'] <= end_dato)]

        # Merge datasettene til et stort et:
        merged_1 = pd.merge(avg_demand_per_day, avg_price_per_day, on='Date')
        merged_1['ID'] = ID
        merged = pd.merge(merged_1, temp_filtered, on='Date')

        filtered = merged[(merged['Avg demand kWh per day'] > 0) & (merged['Avg price NOK kWh per day'] > 0) & (
            merged['Temperatur'].notnull())].copy()

        df = pd.DataFrame(filtered)
        cols = ['ID'] + [col for col in df.columns if col != 'ID']
        df = df[cols]

        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        df['Month'] = df['Date'].dt.strftime('%B')

        # pd.set_option('display.max_colwidth', None)
        # pd.set_option('display.width', None)
        # pd.set_option('display.max_rows', None)
        # print(df)

        # Kjører regresjonsanalysen:

        df['Month'] = pd.Categorical(df['Month'], categories=['January', 'February', 'March', 'April', 'May', 'June',
                                                              'July', 'August', 'September', 'October', 'November',
                                                              'December'], ordered=True)

        y, X = patsy.dmatrices('Q("Avg demand kWh per day") ~ np.log(Q("Avg price NOK kWh per day")) + Temperatur + '
                               'I(Temperatur**2) + I(Temperatur**3) + Temperatur3 + '
                               'C(Month, Treatment(reference = "April")) + C(Month, Treatment(reference = "April")) * Temperatur3',
                               data=df, return_type='dataframe', NA_action='drop')

        model = sm.OLS(y, X).fit()
        print(model.summary())


def log_lin_prisfølsomhet_pris_dag(test_liste_husstander,data_demand,data_price_update,data_households, Blindern_Temperatur_dag):
    start_dato = '2021-12-01'
    end_dato = '2021-12-31'

    for ID in test_liste_husstander:
        # Gjennomsnits demand per dag:
        demand_ID = data_demand[data_demand['ID'] == ID].copy()
        demand_ID['Date'] = pd.to_datetime(demand_ID['Date'])
        demand_filtered = demand_ID[(demand_ID['Date'] >= start_dato) & (demand_ID['Date'] <= end_dato)]

        avg_demand_per_day = demand_filtered.groupby('Date')['Demand_kWh'].sum().reset_index()
        avg_demand_per_day['Avg demand kWh per day'] = avg_demand_per_day['Demand_kWh'] / 24

        # Gjennomsnitss pris per dag:
        price_area = data_households[data_households['ID'] == ID].iloc[0]['Price_area']
        price_data = data_price_update[data_price_update['Price_area'] == price_area].copy()
        price_data['Date'] = pd.to_datetime(price_data['Date'])
        price_filtered = price_data[(price_data['Date'] >= start_dato) & (price_data['Date'] <= end_dato)]

        avg_price_per_day = price_filtered.groupby('Date')['Price_NOK_kWh'].sum().reset_index()
        avg_price_per_day['Avg price NOK kWh per day'] = avg_price_per_day['Price_NOK_kWh'] / 24

        # Temperatur:
        Blindern_Temperatur_dag['Date'] = pd.to_datetime(Blindern_Temperatur_dag['Date'])
        Blindern_Temperatur_dag['Temperatur3'] = Blindern_Temperatur_dag['Temperatur'].rolling(window=3,
                                                                                               min_periods=1).mean()
        temp_filtered = Blindern_Temperatur_dag[
            (Blindern_Temperatur_dag['Date'] >= start_dato) & (Blindern_Temperatur_dag['Date'] <= end_dato)]

        # Merge datasettene til et stort et:
        merged_1 = pd.merge(avg_demand_per_day, avg_price_per_day, on='Date')
        merged_1['ID'] = ID
        merged = pd.merge(merged_1, temp_filtered, on='Date')

        filtered = merged[(merged['Avg demand kWh per day'] > 0) & (merged['Avg price NOK kWh per day'] > 0) & (
            merged['Temperatur'].notnull())].copy()

        df = pd.DataFrame(filtered)
        cols = ['ID'] + [col for col in df.columns if col != 'ID']
        df = df[cols]

        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        df['Month'] = df['Date'].dt.strftime('%B')

        # pd.set_option('display.max_colwidth', None)
        # pd.set_option('display.width', None)
        # pd.set_option('display.max_rows', None)
        # print(df)

        # Kjører regresjonsanalysen:

        df['Month'] = pd.Categorical(df['Month'], categories=['January', 'February', 'March', 'April', 'May', 'June',
                                                              'July', 'August', 'September', 'October', 'November',
                                                              'December'], ordered=True)

        y, X = patsy.dmatrices('np.log(Q("Avg demand kWh per day")) ~ Q("Avg price NOK kWh per day") + Temperatur + '
                               'I(Temperatur**2) + I(Temperatur**3) + Temperatur3 + '
                               'C(Month, Treatment(reference = "April")) + C(Month, Treatment(reference = "April")) * Temperatur3',
                               data=df, return_type='dataframe', NA_action='drop')

        model = sm.OLS(y, X).fit()
        print(model.summary())

#-----------------------------------------------------------------------------------

'''Regresjon for "direkte", ren regresjonsanalyse: demand = beta_0 + beta_1 *pris + beta_2 *T + beta_3 *T^2 + beta_4 *T^3 +
                                                            Temperatur3 + Month + Month*Temperatur3'''

def direkte_prisfølsomhet_dag(test_liste_husstander,data_demand,data_price_update,data_households, Blindern_Temperatur_dag):
    start_dato = '2021-04-01'
    end_dato = '2022-03-31'

    for ID in test_liste_husstander:
        # Gjennomsnits demand per dag:
        demand_ID = data_demand[data_demand['ID'] == ID].copy()
        demand_ID['Date'] = pd.to_datetime(demand_ID['Date'])
        demand_filtered = demand_ID[(demand_ID['Date'] >= start_dato) & (demand_ID['Date'] <= end_dato)]

        avg_demand_per_day = demand_filtered.groupby('Date')['Demand_kWh'].sum().reset_index()
        avg_demand_per_day['Avg demand kWh per day'] = avg_demand_per_day['Demand_kWh'] / 24

        # Gjennomsnitss pris per dag:
        price_area = data_households[data_households['ID'] == ID].iloc[0]['Price_area']
        price_data = data_price_update[data_price_update['Price_area'] == price_area].copy()
        price_data['Date'] = pd.to_datetime(price_data['Date'])
        price_filtered = price_data[(price_data['Date'] >= start_dato) & (price_data['Date'] <= end_dato)]

        avg_price_per_day = price_filtered.groupby('Date')['Price_NOK_kWh'].sum().reset_index()
        avg_price_per_day['Avg price NOK kWh per day'] = avg_price_per_day['Price_NOK_kWh'] / 24

        # Temperatur:
        Blindern_Temperatur_dag['Date'] = pd.to_datetime(Blindern_Temperatur_dag['Date'])
        Blindern_Temperatur_dag['Temperatur3'] = Blindern_Temperatur_dag['Temperatur'].rolling(window=3, min_periods=1).mean()

        temp_filtered = Blindern_Temperatur_dag[
            (Blindern_Temperatur_dag['Date'] >= start_dato) & (Blindern_Temperatur_dag['Date'] <= end_dato)]

        # Merge datasettene til et stort et:
        merged_1 = pd.merge(avg_demand_per_day, avg_price_per_day, on='Date')
        merged_1['ID'] = ID
        merged = pd.merge(merged_1, temp_filtered, on='Date')

        filtered = merged[(merged['Avg demand kWh per day'] > 0) & (merged['Avg price NOK kWh per day'] > 0) & (
            merged['Temperatur'].notnull())].copy()

        df = pd.DataFrame(filtered)
        cols = ['ID'] + [col for col in df.columns if col != 'ID']
        df = df[cols]

        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        df['Month'] = df['Date'].dt.strftime('%B')

        # pd.set_option('display.max_colwidth', None)
        # pd.set_option('display.width', None)
        # pd.set_option('display.max_rows', None)
        # print(df)

        # Kjører regresjonsanalysen:

        df['Month'] = pd.Categorical(df['Month'], categories=['January', 'February', 'March', 'April', 'May', 'June',
                                                              'July', 'August', 'September', 'October', 'November',
                                                              'December'], ordered=True)

        y, X = patsy.dmatrices('Q("Avg demand kWh per day") ~ Q("Avg price NOK kWh per day") + Temperatur + '
                               'I(Temperatur**2) + I(Temperatur**3) + Temperatur3 + '
                               'C(Month, Treatment(reference = "April")) + C(Month, Treatment(reference = "April")) * Temperatur3',
                               data=df, return_type='dataframe', NA_action='drop')

        model = sm.OLS(y, X).fit()
        print(model.summary())

#-----------------------------------------------------------------------------------

'''Kjøre funksjonene, printer ut resultatene '''

#resultater = log_log_prisfølsomhet_dag(test_liste_husstander,data_demand, data_price_update, data_households, Blindern_Temperatur_dag)
#resultater = log_lin_tempfølsomhet_temp_dag(test_liste_husstander,data_demand, data_price_update, data_households, Blindern_Temperatur_dag)
#resultater = lin_log_prisfølsomhet_pris_dag(test_liste_husstander,data_demand,data_price_update,data_households, Blindern_Temperatur_dag)
#resultater = log_lin_prisfølsomhet_pris_dag(test_liste_husstander,data_demand,data_price_update,data_households, Blindern_Temperatur_dag)
resultater = direkte_prisfølsomhet_dag(test_liste_husstander,data_demand,data_price_update,data_households, Blindern_Temperatur_dag)

print(resultater)



