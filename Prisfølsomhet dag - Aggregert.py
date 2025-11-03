import numpy as np
import pandas as pd
import csv
import row

import statsmodels.api as sm


# -------------------------------------- LESER DATA --------------------------------------#

data_demand = pd.read_csv('/Users/kristinemoen/Documents/5-klasse/Prosjektoppgave_CSV_filer/demand.csv')

data_price = pd.read_csv('/Users/kristinemoen/Documents/5-klasse/Prosjektoppgave_CSV_filer/prices.csv')
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

test_liste_husstander = [512, 642] #Bare for test

#-----------------------------------------------------------------------------------

'''Regresjon for log-log: log(demand) = beta_0 + beta_1 *log(pris) + beta_2 *T + beta_3 *T^2 + beta_4 *T^3'''

def log_log_dag(liste_husstander, data_demand, data_price_update, data_households, Blindern_Temperatur_dag):
    resultater = []
    start_dato = pd.to_datetime('2021-04-01')
    end_dato = pd.to_datetime('2022-03-31')

    #Gjennomsnits demand per dag for alle ID-ene:
    data_demand['Date'] = pd.to_datetime(data_demand['Date'])
    demand_data_filtered = data_demand[(data_demand['ID'].isin(liste_husstander)) &
                              (data_demand['Date'] >= start_dato) &
                               (data_demand['Date'] <= end_dato)].copy()

    total_daglig_demand = demand_data_filtered.groupby('Date')['Demand_kWh'].sum().reset_index()
    total_daglig_demand['Avg demand kWh per day'] = total_daglig_demand['Demand_kWh']/24

    #Gjennosnits prisen:
    price_area = data_households[data_households['ID'].isin(liste_husstander)].iloc[0]['Price_area']
    price_data = data_price_update[data_price_update['Price_area'] == price_area].copy()
    price_data['Date'] = pd.to_datetime(price_data['Date'])
    price_filtered = price_data[(price_data['Date'] >= start_dato) & (price_data['Date'] <= end_dato)]

    tot_daglig_pris = price_filtered.groupby('Date')['Price_NOK_kWh'].sum().reset_index()
    tot_daglig_pris['Avg price NOK kWh per day'] = tot_daglig_pris['Price_NOK_kWh']/24

    #Gjennomsnits temoperatur:
    Blindern_Temperatur_dag['Date'] = pd.to_datetime(Blindern_Temperatur_dag['Date'])

    #Merge:
    merged_1 = pd.merge(total_daglig_demand, tot_daglig_pris, on = 'Date')
    merged = pd.merge(merged_1, Blindern_Temperatur_dag, on ='Date')

    filtered = merged[(merged['Avg demand kWh per day'] > 0) & (merged['Avg price NOK kWh per day'] > 0 ) & (merged['Temperatur'].notnull())].copy()

    #Beregninger:
    if len(filtered) > 0:
        filtered['log_demand'] = np.log(filtered['Avg demand kWh per day'])  # Logartitmen av strømprisen
        filtered['log_price'] = np.log(filtered['Avg price NOK kWh per day'])  # Logaritmen av strømforbruket
        filtered['T'] = filtered['Temperatur']
        filtered['T2'] = filtered['T'] ** 2
        filtered['T3'] = filtered['T'] ** 3

        # Regresjonsanalyse: log(Demand) = beta_0 + beta_1 * log(Price) + beta_2 * T + beta_3 *T^2 + beta_4 *T^3 + error
        X = filtered[['log_price', 'T', 'T2', 'T3']]
        X_1 = sm.add_constant(X)                           # Legger til en konstant
        y = filtered['log_demand']                         # Setter opp responsvariabelen
        model = sm.OLS(y, X_1).fit()                       # Kjører en lineær regresjonsanalyse, finner den beste linjen som passer dataene
        beta_log_log = model.params['log_price']           # Forteller om prisfølsomheten

        regresjonslinje_log_log = (f"log(demand) = {model.params['const']: .2f} + "
                                   f"{model.params['log_price']: .2f} *log(price NOK/kWh) + "
                                   f"{model.params['T']: .2f} *T + "
                                   f"{model.params['T2']: .4f} *T^2 + "
                                   f"{model.params['T3']: .4f} *T^3")

        print(model.summary())

        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_rows', None)
        resultater.append({
            'Prisfølsomhet (beta) for log log logarithm': beta_log_log,
            'Regresjonsliste for log log logarithm': regresjonslinje_log_log
        })


    return pd.DataFrame(resultater)

#-----------------------------------------------------------------------------------

'''Regresjon for log-lin/ lin-log: 
      1) log(demand) = beta_0 + beta_1 *T + beta_2 *T^2 + beta_3 *T^3 + beta_4 *pris
      2) demand = beta_0 + beta_1 *log(pris) + beta_2 *T + beta_3 *T^2 + Beta_4 *T^3
      3) log(demand) = beta_0 + beta_1 *pris + beta_2 *T + beta_3 *T^2 + Beta_4 *T^3
'''

def log_lin_temp_dag(liste_husstander,data_demand, data_price_update, data_households, Blindern_Temperatur_dag):
    resultater = []
    start_dato = pd.to_datetime('2021-04-01')
    end_dato = pd.to_datetime('2022-03-31')

    # Gjennomsnits demand per dag for alle ID-ene:
    data_demand['Date'] = pd.to_datetime(data_demand['Date'])
    demand_data_filtered = data_demand[(data_demand['ID'].isin(liste_husstander)) &
                                       (data_demand['Date'] >= start_dato) &
                                       (data_demand['Date'] <= end_dato)].copy()

    total_daglig_demand = demand_data_filtered.groupby('Date')['Demand_kWh'].sum().reset_index()
    total_daglig_demand['Avg demand kWh per day'] = total_daglig_demand['Demand_kWh'] / 24

    # Gjennosnits prisen:
    price_area = data_households[data_households['ID'].isin(liste_husstander)].iloc[0]['Price_area']
    price_data = data_price_update[data_price_update['Price_area'] == price_area].copy()
    price_data['Date'] = pd.to_datetime(price_data['Date'])
    price_filtered = price_data[(price_data['Date'] >= start_dato) & (price_data['Date'] <= end_dato)]

    tot_daglig_pris = price_filtered.groupby('Date')['Price_NOK_kWh'].sum().reset_index()
    tot_daglig_pris['Avg price NOK kWh per day'] = tot_daglig_pris['Price_NOK_kWh'] / 24

    # Gjennomsnits temoperatur:
    Blindern_Temperatur_dag['Date'] = pd.to_datetime(Blindern_Temperatur_dag['Date'])

    # Merge:
    merged_1 = pd.merge(total_daglig_demand, tot_daglig_pris, on='Date')
    merged = pd.merge(merged_1, Blindern_Temperatur_dag, on='Date')

    filtered = merged[(merged['Avg demand kWh per day'] > 0) & (merged['Avg price NOK kWh per day'] > 0) & (merged['Temperatur'].notnull())].copy()

    #Beregninger:
    if len(filtered)>0:
        filtered['log_demand'] = np.log(filtered['Avg demand kWh per day'])  # Logartitmen av strømprisen
        filtered['T'] = filtered['Temperatur']
        filtered['T2'] = filtered['T'] ** 2
        filtered['T3'] = filtered['T'] ** 3
        filtered['price'] = filtered['Avg price NOK kWh per day']

        # Regresjonsanalyse: log(demand) = beta_0 + beta_1 * T + beta_2 * pris + beta_3 *T^2 + beta_3 * T^3 + error
        X_natural = sm.add_constant(filtered[['T', 'T2', 'T3', 'price']])
        y_natural = filtered['log_demand']
        model_natural = sm.OLS(y_natural, X_natural).fit()
        beta_natural = model_natural.params['T']

        regresjonslinje_natural = (f"log(demand) = {model_natural.params['const']: .2f} + "
                                   f"{model_natural.params['T']: .2f} *T +"
                                   f"{model_natural.params['T2']} *T^2 +"
                                   f"{model_natural.params['T3']} *T^3 + "
                                   f" {model_natural.params['price']} *price")
        print(model_natural.summary())

        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_rows', None)
        resultater.append({
            'Prisfølsomhet (beta) for Natural logarithm': beta_natural,
            'Regresjonslinjen for log-lineær': regresjonslinje_natural
        })

    else:
        beta_natural = np.nan
        regresjonslinje_natural = None

    return pd.DataFrame(resultater)

def lin_log_pris_dag(test_liste_husstander, data_demand, data_price_update, data_households, Blindern_Temperatur_dag):
    resultater = []
    start_dato = pd.to_datetime('2021-04-01')
    end_dato = pd.to_datetime('2022-03-31')

    # Gjennomsnits demand per dag for alle ID-ene:
    data_demand['Date'] = pd.to_datetime(data_demand['Date'])
    demand_data_filtered = data_demand[(data_demand['ID'].isin(test_liste_husstander)) &
                                       (data_demand['Date'] >= start_dato) &
                                       (data_demand['Date'] <= end_dato)].copy()

    total_daglig_demand = demand_data_filtered.groupby('Date')['Demand_kWh'].sum().reset_index()
    total_daglig_demand['Avg demand kWh per day'] = total_daglig_demand['Demand_kWh'] / 24

    # Gjennosnits prisen:
    price_area = data_households[data_households['ID'].isin(test_liste_husstander)].iloc[0]['Price_area']
    price_data = data_price_update[data_price_update['Price_area'] == price_area].copy()
    price_data['Date'] = pd.to_datetime(price_data['Date'])
    price_filtered = price_data[(price_data['Date'] >= start_dato) & (price_data['Date'] <= end_dato)]

    tot_daglig_pris = price_filtered.groupby('Date')['Price_NOK_kWh'].sum().reset_index()
    tot_daglig_pris['Avg price NOK kWh per day'] = tot_daglig_pris['Price_NOK_kWh'] / 24

    # Gjennomsnits temoperatur:
    Blindern_Temperatur_dag['Date'] = pd.to_datetime(Blindern_Temperatur_dag['Date'])

    # Merge:
    merged_1 = pd.merge(total_daglig_demand, tot_daglig_pris, on='Date')
    merged = pd.merge(merged_1, Blindern_Temperatur_dag, on='Date')

    filtered = merged[(merged['Avg demand kWh per day'] > 0) & (merged['Avg price NOK kWh per day'] > 0) & (
        merged['Temperatur'].notnull())].copy()

    # Beregninger:
    if len(filtered) > 0:
        # Demand = beta_0 + beta_1 * log(price) + beta_2 * T^2 + beta_3 * T^3
        filtered['log_price'] = np.log(filtered['Avg price NOK kWh per day'])  # Logartitmen av strømprisen
        filtered['demand'] = filtered['Avg demand kWh per day']
        filtered['T'] = filtered['Temperatur']
        filtered['T2'] = filtered['T'] ** 2
        filtered['T3'] = filtered['T'] ** 3

        X_natural = sm.add_constant(filtered[['log_price', 'T', 'T2', 'T3']])
        y_natural = filtered['demand']
        model_natural = sm.OLS(y_natural, X_natural).fit()
        beta_natural = model_natural.params['log_price']

        regresjonslinje_natural = (f"demand = {model_natural.params['const']: .2f} + "
                                   f"{model_natural.params['log_price']: .2f} *log(price) +"
                                   f"{model_natural.params['T']: .3f} *T +"
                                   f"{model_natural.params['T2']: .4f} *T^2 +"
                                   f"{model_natural.params['T3']: .4f} *T^3")
        print(model_natural.summary())

        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_rows', None)


    else:
        beta_natural = np.nan
        regresjonslinje_natural = None

    resultater.append({
        'Prisfølsomhet (beta) for Natural logarithm': beta_natural,
        'Regresjonslinjen for lineær-log': regresjonslinje_natural
    })

    return pd.DataFrame(resultater)

def log_lin_pris_dag(test_liste_husstander,data_demand,data_price_update,data_households, Blindern_Temperatur_dag):
    resultater = []
    start_dato = pd.to_datetime('2021-04-01')
    end_dato = pd.to_datetime('2022-03-31')

    # Gjennomsnits demand per dag for alle ID-ene:
    data_demand['Date'] = pd.to_datetime(data_demand['Date'])
    demand_data_filtered = data_demand[(data_demand['ID'].isin(test_liste_husstander)) &
                                       (data_demand['Date'] >= start_dato) &
                                       (data_demand['Date'] <= end_dato)].copy()

    total_daglig_demand = demand_data_filtered.groupby('Date')['Demand_kWh'].sum().reset_index()
    total_daglig_demand['Avg demand kWh per day'] = total_daglig_demand['Demand_kWh'] / 24

    # Gjennosnits prisen:
    price_area = data_households[data_households['ID'].isin(test_liste_husstander)].iloc[0]['Price_area']
    price_data = data_price_update[data_price_update['Price_area'] == price_area].copy()
    price_data['Date'] = pd.to_datetime(price_data['Date'])
    price_filtered = price_data[(price_data['Date'] >= start_dato) & (price_data['Date'] <= end_dato)]

    tot_daglig_pris = price_filtered.groupby('Date')['Price_NOK_kWh'].sum().reset_index()
    tot_daglig_pris['Avg price NOK kWh per day'] = tot_daglig_pris['Price_NOK_kWh'] / 24

    # Gjennomsnits temoperatur:
    Blindern_Temperatur_dag['Date'] = pd.to_datetime(Blindern_Temperatur_dag['Date'])

    # Merge:
    merged_1 = pd.merge(total_daglig_demand, tot_daglig_pris, on='Date')
    merged = pd.merge(merged_1, Blindern_Temperatur_dag, on='Date')

    filtered = merged[(merged['Avg demand kWh per day'] > 0) & (merged['Avg price NOK kWh per day'] > 0) & (
        merged['Temperatur'].notnull())].copy()

    # Beregninger:
    if len(filtered) > 0:
        # log(demand) = beta_0 + beta_1 * price
        filtered['price'] = filtered['Avg price NOK kWh per day']  # Logartitmen av strømprisen
        filtered['log_demand'] = np.log(filtered['Avg demand kWh per day'])
        filtered['T'] = filtered['Temperatur']
        filtered['T2'] = filtered['T'] ** 2
        filtered['T3'] = filtered['T'] ** 3

        X = filtered[['price', 'T', 'T2', 'T3']]
        X_natural = sm.add_constant(X)
        y_natural = filtered['log_demand']
        model_natural = sm.OLS(y_natural, X_natural).fit()
        beta_natural = model_natural.params['price']

        regresjonslinje_natural = (f"log(demand) = {model_natural.params['const']: .2f} + "
                                   f"{model_natural.params['price']: .2f} *price +"
                                   f"{model_natural.params['T']: .3f} *T +"
                                   f"{model_natural.params['T2']: .4f} *T^2 +"
                                   f"{model_natural.params['T3']: .4f} *T^3")

        print(model_natural.summary())

        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_rows', None)


    else:
        beta_natural = np.nan
        regresjonslinje_natural = None

    resultater.append({
        'Prisfølsomhet (beta) for Natural logarithm': beta_natural,
        'Regresjonslinjen for log-lin': regresjonslinje_natural
    })

    return pd.DataFrame(resultater)

#-----------------------------------------------------------------------------------

'''Regresjon for "direkte", ren regresjonsanalyse: demand = beta_0 + beta_1 *pris + beta_2 *T + beta_3 *T^2 + beta_4 *T^3'''

def direkte_dag(test_liste_husstander,data_demand,data_price_update,data_households, Blindern_Temperatur_dag):
    resultater = []
    start_dato = pd.to_datetime('2021-04-01')
    end_dato = pd.to_datetime('2022-03-31')

    # Gjennomsnits demand per dag for alle ID-ene:
    data_demand['Date'] = pd.to_datetime(data_demand['Date'])
    demand_data_filtered = data_demand[(data_demand['ID'].isin(test_liste_husstander)) &
                                       (data_demand['Date'] >= start_dato) &
                                       (data_demand['Date'] <= end_dato)].copy()

    total_daglig_demand = demand_data_filtered.groupby('Date')['Demand_kWh'].sum().reset_index()
    total_daglig_demand['Avg demand kWh per day'] = total_daglig_demand['Demand_kWh'] / 24

    # Gjennosnits prisen:
    price_area = data_households[data_households['ID'].isin(test_liste_husstander)].iloc[0]['Price_area']
    price_data = data_price_update[data_price_update['Price_area'] == price_area].copy()
    price_data['Date'] = pd.to_datetime(price_data['Date'])
    price_filtered = price_data[(price_data['Date'] >= start_dato) & (price_data['Date'] <= end_dato)]

    tot_daglig_pris = price_filtered.groupby('Date')['Price_NOK_kWh'].sum().reset_index()
    tot_daglig_pris['Avg price NOK kWh per day'] = tot_daglig_pris['Price_NOK_kWh'] / 24

    # Gjennomsnits temoperatur:
    Blindern_Temperatur_dag['Date'] = pd.to_datetime(Blindern_Temperatur_dag['Date'])

    # Merge:
    merged_1 = pd.merge(total_daglig_demand, tot_daglig_pris, on='Date')
    merged = pd.merge(merged_1, Blindern_Temperatur_dag, on='Date')

    filtered = merged[(merged['Avg demand kWh per day'] > 0) & (merged['Avg price NOK kWh per day'] > 0) & (
        merged['Temperatur'].notnull())].copy()

    # Beregninger:
    if len(filtered) > 0:
        # Regresjonslinje: Demand = beta_0 + beta_1 * pris + beta_2 * T + beta_3 *T^2 + beta_4 *T^3
        filtered['price'] = filtered['Avg price NOK kWh per day']  # Logartitmen av strømprisen
        filtered['demand'] = np.log(filtered['Avg demand kWh per day'])
        filtered['T'] = filtered['Temperatur']
        filtered['T2'] = filtered['T'] ** 2
        filtered['T3'] = filtered['T'] ** 3

        X_natural = filtered[['price', 'T', 'T2', 'T3']]
        X_natural = sm.add_constant(X_natural)
        y_natural = filtered['demand']
        model_natural = sm.OLS(y_natural, X_natural).fit()
        beta_natural = model_natural.params['price']

        regresjonslinje_natural = (f"demand = {model_natural.params['const']: .2f} + "
                                   f"{model_natural.params['price']: .2f} *price + "
                                   f"{model_natural.params['T']: .4f} *T + "
                                   f"{model_natural.params['T2']: .4f} *T^2 + "
                                   f"{model_natural.params['T3']: .4f} *T^3")
        print(model_natural.summary())

    else:
        beta_natural = np.nan
        regresjonslinje_natural = None

    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_rows', None)
    resultater.append({
        'Prisfølsomhet (beta) for lineær regresjonsanalyse': beta_natural,
        'Regresjonslinjen for lineær': regresjonslinje_natural
    })

    return pd.DataFrame(resultater)


#-----------------------------------------------------------------------------------

'''Kjøre funksjonene, printer ut resultatene '''


#resultater = log_log_dag(liste_husstander,data_demand,data_price_update,data_households,Blindern_Temperatur_dag)
#resultater = log_lin_temp_dag(liste_husstander,data_demand, data_price_update, data_households, Blindern_Temperatur_dag)
#resultater = lin_log_pris_dag(test_liste_husstander, data_demand, data_price_update, data_households, Blindern_Temperatur_dag)
#resultater = lin_log_pris_dag(test_liste_husstander, data_demand, data_price_update, data_households, Blindern_Temperatur_dag)
resultater = direkte_dag(test_liste_husstander,data_demand,data_price_update,data_households, Blindern_Temperatur_dag)


print(resultater)

