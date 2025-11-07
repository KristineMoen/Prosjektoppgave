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

data_price = pd.read_csv('/Users/kristinemoen/Documents/5-klasse/Prosjektoppgave_CSV_filer/prices.csv')
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

#--------------------------------- REGNE PÅ PRISFØLSOMHET PER TIME FOR TIME ------------------------------------------#

test_liste_husstander = [512, 642, 827] #Bare for test

#-----------------------------------------------------------------------------------

'''Regresjon for "direkte", ren regresjonsanalyse: demand = beta_0 + beta_1 *pris + beta_2 *T + beta_3 *T^2 + beta_4 *T^3 + sum(alpha_h *time_h) + error'''

def direkte_prisfølsomhet_time(test_liste_husstander, data_demand, data_price_update, data_households, Blindern_Temp_t4t):
    resultater = []
    start_dato = '2021-12-01'
    end_dato ='2021-12-31'

    for ID in test_liste_husstander:
        # demand per time:
        demand_ID = data_demand[data_demand['ID'] == ID].copy()
        demand_ID['Date'] = pd.to_datetime(demand_ID['Date'])
        demand_ID['Hour'] = demand_ID['Hour'].astype(int)
        demand_filtered = demand_ID[(demand_ID['Date'] >= start_dato) & (demand_ID['Date'] <= end_dato)]


        # pris per time:
        price_area = data_households[data_households['ID'] == ID].iloc[0]['Price_area']
        price_data = data_price_update[data_price_update['Price_area'] == price_area].copy()
        price_data['Date'] = pd.to_datetime(price_data['Date'])
        price_data['Hour'] = price_data['Hour'].astype(int)
        price_filtered = price_data[(price_data['Date'] >= start_dato) & (price_data['Date'] <= end_dato)]

        # Temperatur:
        Blindern_Temp_t4t['Date'] = pd.to_datetime(Blindern_Temp_t4t['Date'])
        Blindern_Temp_t4t['Hour'] = Blindern_Temp_t4t['Hour'].astype(int)
        temp_filtered = Blindern_Temp_t4t[
            (Blindern_Temp_t4t['Date'] >= start_dato) & (Blindern_Temp_t4t['Date'] <= end_dato)]

        temp_filtered.loc[:, 'Temperatur'] = temp_filtered['Temperatur']


        # Merge datasettene til et stort:
        merged_1 = pd.merge(demand_filtered, price_filtered, on=['Date', 'Hour'])
        merged_1['ID'] = ID
        merged = pd.merge(merged_1, Blindern_Temp_t4t, on=['Date', 'Hour'])

        filtered = merged[(merged['Demand_kWh'] > 0) & (merged['Price_NOK_kWh'] > 0) & (merged['Temperatur'].notnull())].copy()

        if len(filtered) > 0:
            filtered['demand'] = filtered['Demand_kWh']  # Logaritmen av strømforbruket
            filtered['price'] = filtered['Price_NOK_kWh']  # Logartitmen av strømprisen
            filtered['T'] = filtered['Temperatur']
            filtered['T2'] = filtered['T'] ** 2
            filtered['T3'] = filtered['T'] ** 3
            filtered['hour'] = filtered['Hour']

            hour_dummies = pd.get_dummies(filtered['hour'], prefix='hour')


            # Regresjonsanalyse: demand = beta_0 + beta_1 * pris + beta_2 *T + beta_3 *T^2 + beta_4 *T^3 + sum(alpha_h *time_h) + error
            X_natural = pd.concat([filtered[['price', 'T', 'T2', 'T3']], hour_dummies], axis = 1)
            X_natural = sm.add_constant(X_natural)
            X_natural = X_natural.astype(float)
            y_natural = filtered['demand'].astype(float)

            model_natural = sm.OLS(y_natural, X_natural).fit()
            beta_natural = model_natural.params['price']
            hour_params = {param: value for param, value in model_natural.params.items() if param.startswith('hour')}
            hour_str = " + ".join([f"{v: .2f} * {k}" for k, v in hour_params.items()])

            regresjonslinje_linear = (f"demand = {model_natural.params['const']: .2f} +"
                                      f" {model_natural.params['price']: .2f} *price + "
                                      f"{model_natural.params['T']: .2f} * T + "
                                      f"{model_natural.params['T2']: .2f} * T^2 + "
                                      f"{model_natural.params['T3']: .2f} * T^3 " +
                                      hour_str)
            print('For ID: ' + str(ID))
            print(model_natural.summary())

            full_regresjonslinje = "demand = "
            for param, value in model_natural.params.items():
                full_regresjonslinje += f"{value: .2f} * {param} + "
            full_regresjonslinje = full_regresjonslinje.rstrip(" + ")
            print(full_regresjonslinje)


        else:
            beta_natural = np.nan
            regresjonslinje_linear = None

        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_rows', None)
        resultater.append({
            'ID': ID,
            'Prisfølsomhet (beta) for linear regresjonsmodell': beta_natural,
            'Regresjonslinjen for linear regresjonsmodell': regresjonslinje_linear
        })

    return pd.DataFrame(resultater)

#-----------------------------------------------------------------------------------
'''Regresjon for log-lin/ lin-log: 
      1) log(demand) = beta_0 + beta_1 *T + beta_2 *T^2 + beta_3 *T^3 + beta_4 *pris + sum(alpha_h *time_h) + error
      2) demand = beta_0 + beta_1 *log(pris) + beta_2 *T + beta_3 *T^2 + Beta_4 *T^3 + sum(alpha_h *time_h) + error
      3) log(demand) = beta_0 + beta_1 *pris + beta_2 *T + beta_3 *T^2 + Beta_4 *T^3 + sum(alpha_h *time_h) + error
'''

def log_lin_prisfølsomhet_temp_time(test_liste_husstander,data_demand, data_price_update, data_households, Blindern_Temp_t4t):
    resultater = []
    start_dato = '2021-12-01'
    end_dato = '2021-12-02'
    start_dato = pd.to_datetime(start_dato)
    end_dato = pd.to_datetime(end_dato)

    for ID in test_liste_husstander:
        # demand per time:
        demand_ID = data_demand[data_demand['ID'] == ID].copy()
        demand_ID['Date'] = pd.to_datetime(demand_ID['Date'])
        demand_ID['Hour'] = demand_ID['Hour'].astype(int)
        demand_filtered = demand_ID[(demand_ID['Date'] >= start_dato) & (demand_ID['Date'] <= end_dato)]

        demand_filtered.loc[:, 'Demand kWh per hour'] = demand_filtered['Demand_kWh']

        # pris per time:
        price_area = data_households[data_households['ID'] == ID].iloc[0]['Price_area']
        price_data = data_price_update[data_price_update['Price_area'] == price_area].copy()
        price_data['Date'] = pd.to_datetime(price_data['Date'])
        price_data['Hour'] = price_data['Hour'].astype(int)
        price_filtered = price_data[(price_data['Date'] >= start_dato) & (price_data['Date'] <= end_dato)]

        price_filtered.loc[:, 'Price NOK kWh per hour'] = price_filtered['Price_NOK_kWh']

        # Temperatur:
        Blindern_Temp_t4t['Date'] = pd.to_datetime(Blindern_Temp_t4t['Date'])
        Blindern_Temp_t4t['Hour'] = Blindern_Temp_t4t['Hour'].astype(int)
        temp_filtered = Blindern_Temp_t4t[(Blindern_Temp_t4t['Date'] >= start_dato) & (Blindern_Temp_t4t['Date'] <= end_dato)]

        temp_filtered.loc[:, 'Temperatur'] = temp_filtered['Temperatur']

        # Merge datasettene til et stort:
        merged_1 = pd.merge(demand_filtered, price_filtered, on=['Date','Hour'])
        merged_1['ID'] = ID
        merged = pd.merge(merged_1, Blindern_Temp_t4t, on=['Date','Hour'])

        filtered = merged[(merged['Demand_kWh'] > 0) & (merged['Price_NOK_kWh'] > 0) & (merged['Temperatur'].notnull())].copy()


        if len(filtered) > 0:
            filtered['log_demand'] = np.log(filtered['Demand kWh per hour'])  # Logartitmen av strømprisen
            filtered['T'] = filtered['Temperatur']
            filtered['T2'] = filtered['T'] ** 2
            filtered['T3'] = filtered['T'] ** 3
            filtered['price'] = filtered['Price NOK kWh per hour']
            filtered['hour'] = filtered['Hour']

            hour_dummies = pd.get_dummies(filtered['hour'], prefix='hour')

            X = pd.concat([filtered[['T', 'T2', 'T3', 'price']], hour_dummies], axis = 1)
            X = sm.add_constant(X)
            X = X.astype(float)
            y = filtered['log_demand'].astype(float)

            model = sm.OLS(y,X).fit()
            beta = model.params['T']

            hour_params = {param: value for param, value in model.params.items() if param.startswith('hour')}
            hour_str = " + ".join([f"{v: .2f} * {k}" for k, v in hour_params.items()])

            regresjonslinje = (f"log(demand) = {model.params['const'] : .2f} + "
                               f"{model.params['T']: .2f} *T +"
                               f"{model.params['T2']: .2f} *T^2 +"
                               f"{model.params['T3']: .2f} *T^3 " +
                               hour_str)
            print(f"For ID: {ID}")
            print(model.summary())

        else:
            beta = np.nan
            regresjonslinje = None

        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_rows', None)

        resultater.append({
            'ID': ID,
            'Beta': beta,
            'Regresjonslinje': regresjonslinje
        })


    return pd.DataFrame(resultater)


def lin_log_prisfølsomhet_pris_dag(test_liste_husstander, data_demand, data_price_update, data_households, Blindern_Temp_t4t):
    resultater = []
    start_dato = '2021-12-01'
    end_dato = '2021-12-02'
    start_dato = pd.to_datetime(start_dato)
    end_dato = pd.to_datetime(end_dato)

    for ID in test_liste_husstander:
        # demand per time:
        demand_ID = data_demand[data_demand['ID'] == ID].copy()
        demand_ID['Date'] = pd.to_datetime(demand_ID['Date'])
        demand_ID['Hour'] = demand_ID['Hour'].astype(int)
        demand_filtered = demand_ID[(demand_ID['Date'] >= start_dato) & (demand_ID['Date'] <= end_dato)]

        demand_filtered.loc[:, 'Demand kWh per hour'] = demand_filtered['Demand_kWh']

        # pris per time:
        price_area = data_households[data_households['ID'] == ID].iloc[0]['Price_area']
        price_data = data_price_update[data_price_update['Price_area'] == price_area].copy()
        price_data['Date'] = pd.to_datetime(price_data['Date'])
        price_data['Hour'] = price_data['Hour'].astype(int)
        price_filtered = price_data[(price_data['Date'] >= start_dato) & (price_data['Date'] <= end_dato)]

        price_filtered.loc[:, 'Price NOK kWh per hour'] = price_filtered['Price_NOK_kWh']

        # Temperatur:
        Blindern_Temp_t4t['Date'] = pd.to_datetime(Blindern_Temp_t4t['Date'])
        Blindern_Temp_t4t['Hour'] = Blindern_Temp_t4t['Hour'].astype(int)
        temp_filtered = Blindern_Temp_t4t[
            (Blindern_Temp_t4t['Date'] >= start_dato) & (Blindern_Temp_t4t['Date'] <= end_dato)]

        temp_filtered.loc[:, 'Temperatur'] = temp_filtered['Temperatur']

        # Merge datasettene til et stort:
        merged_1 = pd.merge(demand_filtered, price_filtered, on=['Date', 'Hour'])
        merged_1['ID'] = ID
        merged = pd.merge(merged_1, Blindern_Temp_t4t, on=['Date', 'Hour'])

        filtered = merged[
            (merged['Demand_kWh'] > 0) & (merged['Price_NOK_kWh'] > 0) & (merged['Temperatur'].notnull())].copy()

        if len(filtered) > 0:
            #Demand = beta_0 + beta_1 * log(price) + beta_2 * T^2 + beta_3 * T^3
            filtered['log_price'] = np.log(filtered['Price NOK kWh per hour'])  # Logartitmen av strømprisen
            filtered['demand'] = filtered['Demand kWh per hour']
            filtered['T'] = filtered['Temperatur']
            filtered['T2'] = filtered['T'] **2
            filtered['T3'] = filtered['T'] **3
            filtered['hour'] = filtered['Hour']

            hour_dummies = pd.get_dummies(filtered['hour'], prefix='hour')

            X = pd.concat([filtered[['log_price', 'T', 'T2', 'T3']], hour_dummies], axis=1)
            X = sm.add_constant(X)
            X = X.astype(float)
            y = filtered['demand'].astype(float)

            model = sm.OLS(y, X).fit()
            beta = model.params['log_price']

            hour_params = {param: value for param, value in model.params.items() if param.startswith('hour')}
            hour_str = " + ".join([f"{v: .2f} * {k}" for k, v in hour_params.items()])

            regresjonslinje_natural = (f"demand = {model.params['const']: .2f} + "
                                       f"{model.params['log_price']: .2f} *log(price) +"
                                       f"{model.params['T']: .3f} *T +"
                                       f"{model.params['T2']: .4f} *T^2 +"
                                       f"{model.params['T3']: .4f} *T^3" +
                                       hour_str)
            print('For ID: ' + str(ID))
            print(model.summary())

        else:
            beta = np.nan
            regresjonslinje_natural = None

        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_rows', None)
        resultater.append({
            'ID': ID,
            'Prisfølsomhet (beta) for Natural logarithm': beta,
            'Regresjonslinjen for lineær-log': regresjonslinje_natural
        })

    return pd.DataFrame(resultater)


def log_lin_prisfølsomhet_pris_dag(test_liste_husstander,data_demand,data_price_update,data_households, Blindern_Temp_t4t):
    resultater = []
    start_dato = '2021-12-01'
    end_dato = '2021-12-02'
    start_dato = pd.to_datetime(start_dato)
    end_dato = pd.to_datetime(end_dato)

    for ID in test_liste_husstander:
        # demand per time:
        demand_ID = data_demand[data_demand['ID'] == ID].copy()
        demand_ID['Date'] = pd.to_datetime(demand_ID['Date'])
        demand_ID['Hour'] = demand_ID['Hour'].astype(int)
        demand_filtered = demand_ID[(demand_ID['Date'] >= start_dato) & (demand_ID['Date'] <= end_dato)]

        demand_filtered.loc[:, 'Demand kWh per hour'] = demand_filtered['Demand_kWh']

        # pris per time:
        price_area = data_households[data_households['ID'] == ID].iloc[0]['Price_area']
        price_data = data_price_update[data_price_update['Price_area'] == price_area].copy()
        price_data['Date'] = pd.to_datetime(price_data['Date'])
        price_data['Hour'] = price_data['Hour'].astype(int)
        price_filtered = price_data[(price_data['Date'] >= start_dato) & (price_data['Date'] <= end_dato)]

        price_filtered.loc[:, 'Price NOK kWh per hour'] = price_filtered['Price_NOK_kWh']

        # Temperatur:
        Blindern_Temp_t4t['Date'] = pd.to_datetime(Blindern_Temp_t4t['Date'])
        Blindern_Temp_t4t['Hour'] = Blindern_Temp_t4t['Hour'].astype(int)
        temp_filtered = Blindern_Temp_t4t[
            (Blindern_Temp_t4t['Date'] >= start_dato) & (Blindern_Temp_t4t['Date'] <= end_dato)]

        temp_filtered.loc[:, 'Temperatur'] = temp_filtered['Temperatur']

        # Merge datasettene til et stort:
        merged_1 = pd.merge(demand_filtered, price_filtered, on=['Date', 'Hour'])
        merged_1['ID'] = ID
        merged = pd.merge(merged_1, Blindern_Temp_t4t, on=['Date', 'Hour'])

        filtered = merged[
            (merged['Demand_kWh'] > 0) & (merged['Price_NOK_kWh'] > 0) & (merged['Temperatur'].notnull())].copy()

        if len(filtered) > 0:
            # log(demand) = beta_0 + beta_1 * price
            filtered['price'] = filtered['Price NOK kWh per hour']  # Logartitmen av strømprisen
            filtered['log_demand'] = np.log(filtered['Demand kWh per hour'])
            filtered['T'] = filtered['Temperatur']
            filtered['T2'] = filtered['T'] **2
            filtered['T3'] = filtered['T'] **3
            filtered['hour'] = filtered['Hour']

            hour_dummies = pd.get_dummies(filtered['hour'], prefix='hour')

            X = pd.concat([filtered[['price', 'T', 'T2', 'T3']], hour_dummies], axis=1)
            X = sm.add_constant(X)
            X = X.astype(float)
            y = filtered['log_demand'].astype(float)

            model = sm.OLS(y, X).fit()
            beta = model.params['price']

            hour_params = {param: value for param, value in model.params.items() if param.startswith('hour')}
            hour_str = " + ".join([f"{v: .2f} * {k}" for k, v in hour_params.items()])

            regresjonslinje_natural = (f"log(demand) = {model.params['const']: .2f} + "
                                       f"{model.params['price']: .2f} *price +"
                                       f"{model.params['T']: .3f} *T +"
                                       f"{model.params['T2']: .4f} *T^2 +"
                                       f"{model.params['T3']: .4f} *T^3" +
                                       hour_str)
            print('For ID: ' + str(ID))
            print(model.summary())

        else:
            beta = np.nan
            regresjonslinje_natural = None

        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_rows', None)
        resultater.append({
            'ID': ID,
            'Prisfølsomhet (beta) for Natural logarithm': beta,
            'Regresjonslinjen for log-lin': regresjonslinje_natural
        })

    return pd.DataFrame(resultater)

#-----------------------------------------------------------------------------------

'''Regresjon for log-log: log(demand) = beta_0 + beta_1 *log(pris) + beta_2 *T + beta_3 *T^2 + beta_4 *T^3 + sum(alpha_h *time_h) + error'''

def log_log_prisfølsomhet_dag(test_liste_husstander, data_demand, data_price_update, data_households, Blindern_Temp_t4t):
    resultater = []
    start_dato = '2021-12-01'
    end_dato = '2021-12-02'
    start_dato = pd.to_datetime(start_dato)
    end_dato = pd.to_datetime(end_dato)

    for ID in test_liste_husstander:
        # demand per time:
        demand_ID = data_demand[data_demand['ID'] == ID].copy()
        demand_ID['Date'] = pd.to_datetime(demand_ID['Date'])
        demand_ID['Hour'] = demand_ID['Hour'].astype(int)
        demand_filtered = demand_ID[(demand_ID['Date'] >= start_dato) & (demand_ID['Date'] <= end_dato)]

        demand_filtered.loc[:, 'Demand kWh per hour'] = demand_filtered['Demand_kWh']

        # pris per time:
        price_area = data_households[data_households['ID'] == ID].iloc[0]['Price_area']
        price_data = data_price_update[data_price_update['Price_area'] == price_area].copy()
        price_data['Date'] = pd.to_datetime(price_data['Date'])
        price_data['Hour'] = price_data['Hour'].astype(int)
        price_filtered = price_data[(price_data['Date'] >= start_dato) & (price_data['Date'] <= end_dato)]

        price_filtered.loc[:, 'Price NOK kWh per hour'] = price_filtered['Price_NOK_kWh']

        # Temperatur:
        Blindern_Temp_t4t['Date'] = pd.to_datetime(Blindern_Temp_t4t['Date'])
        Blindern_Temp_t4t['Hour'] = Blindern_Temp_t4t['Hour'].astype(int)
        temp_filtered = Blindern_Temp_t4t[
            (Blindern_Temp_t4t['Date'] >= start_dato) & (Blindern_Temp_t4t['Date'] <= end_dato)]

        temp_filtered.loc[:, 'Temperatur'] = temp_filtered['Temperatur']

        # Merge datasettene til et stort:
        merged_1 = pd.merge(demand_filtered, price_filtered, on=['Date', 'Hour'])
        merged_1['ID'] = ID
        merged = pd.merge(merged_1, Blindern_Temp_t4t, on=['Date', 'Hour'])

        filtered = merged[
            (merged['Demand_kWh'] > 0) & (merged['Price_NOK_kWh'] > 0) & (merged['Temperatur'].notnull())].copy()

        if len(filtered) > 0:
            filtered['log_demand'] = np.log(filtered['Demand kWh per hour'])                  # Logartitmen av strømprisen
            filtered['log_price'] = np.log(filtered['Price NOK kWh per hour'])                # Logaritmen av strømforbruket
            filtered['T'] = filtered['Temperatur']
            filtered['T2'] = filtered['T'] ** 2
            filtered['T3'] = filtered['T'] ** 3
            filtered['Chour'] = filtered['Hour']

            hour_dummies = pd.get_dummies(filtered['Chour'], prefix='hour')


            # Regresjonsanalyse: log(Demand) = beta_0 + beta_1 * log(Price) + beta_2 * T + beta_3 *T^2 + beta_4 *T^3 + error
            X = pd.concat([filtered[['log_price', 'T', 'T2', 'T3']], hour_dummies], axis=1)
            X = sm.add_constant(X).astype(float)                                                      # Legger til en konstant
            y = filtered['log_demand'].astype(float)                                                  # Setter opp responsvariabelen
            model = sm.OLS(y, X).fit()                                                  # Kjører en lineær regresjonsanalyse, finner den beste linjen som passer dataene
            beta_log_log = model.params['log_price']                                      # Forteller om prisfølsomheten

            hour_params = {param: value for param, value in model.params.items() if param.startswith('hour')}
            hour_str = " + ".join([f"{v: .2f} * {k}" for k, v in hour_params.items()])

            regresjonslinje_log_log = (f"log(demand) = {model.params['const']: .2f} + "
                                       f"{model.params['log_price']: .2f} *log(price NOK/kWh) + "
                                       f"{model.params['T']: .2f} *T + "
                                       f"{model.params['T2']: .4f} *T^2 + "
                                       f"{model.params['T3']: .4f} *T^3" +
                                       hour_str)

            print('For ID: ' + str(ID))
            print(model.summary())
            print(filtered)
        else:
            beta_log_log = np.nan  # Ikke nok data
            regresjonslinje_log_log = None

        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_rows', None)
        resultater.append({
            'ID': ID,
            'Prisfølsomhet (beta) for log log logarithm': beta_log_log,
            'Regresjonsliste for log log logarithm': regresjonslinje_log_log
        })

    return pd.DataFrame(resultater)

#-----------------------------------------------------------------------------------

'''Kjøre funksjonene, printer ut resultatene '''

#resultater = direkte_prisfølsomhet_time(test_liste_husstander,data_demand,data_price_update, data_households, Blindern_Temp_t4t)
#resultater = log_lin_prisfølsomhet_temp_time(test_liste_husstander,data_demand, data_price_update, data_households, Blindern_Temp_t4t)
#resultater = lin_log_prisfølsomhet_pris_dag(test_liste_husstander, data_demand, data_price_update, data_households, Blindern_Temp_t4t)
#resultater = log_lin_prisfølsomhet_pris_dag(test_liste_husstander,data_demand,data_price_update,data_households, Blindern_Temp_t4t)
resultater = log_log_prisfølsomhet_dag(test_liste_husstander, data_demand, data_price_update, data_households, Blindern_Temp_t4t)

print(resultater)


