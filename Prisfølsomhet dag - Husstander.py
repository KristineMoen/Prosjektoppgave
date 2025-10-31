import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import row

import statsmodels.api as sm
import mplcursors

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

test_liste_husstander = [512, 642, 827] #Bare for test

#-----------------------------------------------------------------------------------

'''Regresjon for log-log: log(demand) = beta_0 + beta_1 *log(pris) + beta_2 *T + beta_3 *T^2 + beta_4 *T^3'''

def log_log_prisfølsomhet_dag(test_liste_husstander, data_demand, data_price_update, data_households, Blindern_Temperatur_dag):
    resultater = []
    start_dato = '2021-04-01'
    end_dato = '2022-03-31'

    avg_demand_per_day =[]
    avg_price_per_day = []
    alle_husholdninger = []

    for ID in test_liste_husstander:
        #Gjennomsnits demand per dag:
        demand_ID = data_demand[data_demand['ID'] == ID].copy()
        demand_ID['Date'] = pd.to_datetime(demand_ID['Date'])
        demand_filtered = demand_ID[(demand_ID['Date'] >= start_dato) & (demand_ID['Date'] <= end_dato)]

        avg_demand_per_day = demand_filtered.groupby('Date')['Demand_kWh'].sum().reset_index()
        avg_demand_per_day['Avg demand kWh per day'] = avg_demand_per_day['Demand_kWh']/24

        avg_demand_per_day['ID'] = ID

        #Gjennomsnitss pris per dag:
        price_area = data_households[data_households['ID'] == ID].iloc[0]['Price_area']
        price_data = data_price_update[data_price_update['Price_area'] == price_area].copy()
        price_data['Date'] = pd.to_datetime(price_data['Date'])
        price_filtered = price_data[(price_data['Date'] >= start_dato) & (price_data['Date'] <= end_dato)]

        avg_price_per_day = price_filtered.groupby('Date')['Price_NOK_kWh'].sum().reset_index()
        avg_price_per_day['Avg price NOK kWh per day'] = avg_price_per_day['Price_NOK_kWh']/24

        #Temperatur:
        Blindern_Temperatur_dag['Date'] = pd.to_datetime(Blindern_Temperatur_dag['Date'])

        #Merge datasettene til et stort et:
        merged_1 = pd.merge(avg_demand_per_day, avg_price_per_day, on = 'Date')
        merged_1['ID'] = ID
        merged = pd.merge(merged_1, Blindern_Temperatur_dag, on = 'Date')


        filtered = merged[(merged['Avg demand kWh per day'] > 0) & (merged['Avg price NOK kWh per day'] > 0) & (merged['Temperatur'].notnull())].copy()

        if len(filtered) > 0:
            filtered['log_demand'] = np.log(filtered['Avg demand kWh per day'])                  # Logartitmen av strømprisen
            filtered['log_price'] = np.log(filtered['Avg price NOK kWh per day'])                # Logaritmen av strømforbruket
            filtered['T'] = filtered['Temperatur']
            filtered['T2'] = filtered['T'] ** 2
            filtered['T3'] = filtered['T'] ** 3


            # Regresjonsanalyse: log(Demand) = beta_0 + beta_1 * log(Price) + beta_2 * T + beta_3 *T^2 + beta_4 *T^3 + error
            X = filtered[['log_price', 'T', 'T2', 'T3']]
            X_1 = sm.add_constant(X)             # Legger til en konstant
            y = filtered['log_demand']                             # Setter opp responsvariabelen
            model = sm.OLS(y, X_1).fit()                        # Kjører en lineær regresjonsanalyse, finner den beste linjen som passer dataene
            beta_log_log = model.params['log_price']                       # Forteller om prisfølsomheten

            regresjonslinje_log_log = (f"log(demand) = {model.params['const']: .2f} + "
                                       f"{model.params['log_price']: .2f} *log(price NOK/kWh) + "
                                       f"{model.params['T']: .2f} *T + "
                                       f"{model.params['T2']: .4f} *T^2 + "
                                       f"{model.params['T3']: .4f} *T^3")

            print('For ID: ' + str(ID))
            print(model.summary())
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

'''Regresjon for log-lin/ lin-log: 
      1) log(demand) = beta_0 + beta_1 *T + beta_2 *T^2 + beta_3 *T^3 + beta_4 *pris
      2) demand = beta_0 + beta_1 *log(pris) + beta_2 *T + beta_3 *T^2 + Beta_4 *T^3
      3) log(demand) = beta_0 + beta_1 *pris + beta_2 *T + beta_3 *T^2 + Beta_4 *T^3
'''

def log_lin_prisfølsomhet_temp_dag(test_liste_husstander,data_demand, data_price_update, data_households, Blindern_Temperatur_dag):
    resultater = []
    start_dato = '2021-04-01'
    end_dato = '2022-03-31'

    for ID in test_liste_husstander:
        #Gjennomsnits demand per dag:
        demand_ID = data_demand[data_demand['ID'] == ID].copy()
        demand_ID['Date'] = pd.to_datetime(demand_ID['Date'])
        demand_filtered = demand_ID[(demand_ID['Date'] >= start_dato) & (demand_ID['Date'] <= end_dato)]

        avg_demand_per_day = demand_filtered.groupby('Date')['Demand_kWh'].sum().reset_index()
        avg_demand_per_day['Avg demand kWh per day'] = avg_demand_per_day['Demand_kWh'] / 24

        avg_demand_per_day['ID'] = ID

        #Gjennomsnitss pris per dag:
        price_area = data_households[data_households['ID'] == ID].iloc[0]['Price_area']
        price_data = data_price_update[data_price_update['Price_area'] == price_area].copy()
        price_data['Date'] = pd.to_datetime(price_data['Date'])
        price_filtered = price_data[(price_data['Date'] >= start_dato) & (price_data['Date'] <= end_dato)]

        avg_price_per_day = price_filtered.groupby('Date')['Price_NOK_kWh'].sum().reset_index()
        avg_price_per_day['Avg price NOK kWh per day'] = avg_price_per_day['Price_NOK_kWh'] / 24

        #Temperatur:
        Blindern_Temperatur_dag['Date'] = pd.to_datetime(Blindern_Temperatur_dag['Date'])

        #Merge datasettene til et stort et:
        merged_1 = pd.merge(avg_demand_per_day, avg_price_per_day, on='Date')
        merged_1['ID'] = ID
        merged = pd.merge(merged_1, Blindern_Temperatur_dag, on='Date')

        filtered = merged[(merged['Avg demand kWh per day'] > 0) & (merged['Avg price NOK kWh per day'] > 0) & (merged['Temperatur'].notnull())].copy()

        if len(filtered) > 0:
            filtered['log_demand'] = np.log(filtered['Avg demand kWh per day'])  # Logartitmen av strømprisen
            filtered['T'] = filtered['Temperatur']
            filtered['T2'] = filtered['T'] ** 2
            filtered['T3'] = filtered['T'] ** 3
            filtered['price'] = filtered['Avg price NOK kWh per day']

            #Regresjonsanalyse: log(demand) = beta_0 + beta_1 * T + beta_2 * pris + beta_3 *T^2 + beta_3 * T^3 + error
            X_natural = sm.add_constant(filtered[['T','T2', 'T3', 'price']])
            y_natural = filtered['log_demand']
            model_natural = sm.OLS(y_natural, X_natural).fit()
            beta_natural = model_natural.params['T']

            regresjonslinje_natural = (f"log(demand) = {model_natural.params['const']: .2f} + "
                                       f"{model_natural.params['T']: .2f} *T +"
                                       f"{model_natural.params['T2']} *T^2 +"
                                       f"{model_natural.params['T3']} *T^3 + "
                                       f" {model_natural.params['price']} *price")
            print('For ID: ' + str(ID))
            print(model_natural.summary())


            # Plot:
            '''plt.figure(figsize=(10, 6))
            plt.scatter(filtered['T'], filtered['log_demand'], alpha=0.5, label='Observasjonspunkt')
            plt.plot(filtered['T'], model_natural.predict(X_natural), color='red', label='Regresjonslinje')
            plt.xlabel('Temperatur')
            plt.ylabel('log(demand kWh)')
            plt.title(f'Prisfølsomhet for strømforbruk (log-lin) for husholdning ID {ID} per dag')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            cursor = mplcursors.cursor(plt.scatter(filtered['T'], filtered['log_demand'], alpha=0.5, label='Observasjonspunkt'), hover = True)
            datoer = filtered['Date'].dt.strftime('%Y-%m-%d').tolist()

            @cursor.connect("add")
            def on_add(sel):
                sel.annotation.set_text(datoer[sel.index])
            plt.show()'''

        else:
            beta_natural = np.nan
            regresjonslinje_natural = None

        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_rows', None)
        resultater.append({
            'ID': ID,
            'Prisfølsomhet (beta) for Natural logarithm': beta_natural,
            'Regresjonslinjen for log-lineær': regresjonslinje_natural
        })

    return pd.DataFrame(resultater)

def lin_log_prisfølsomhet_pris_dag(test_liste_husstander, data_demand, data_price_update, data_housholds, Blindern_Temperatur_dag):
    resultater = []
    start_dato = '2021-12-01'
    end_dato = '2021-12-31'

    for ID in test_liste_husstander:
        #Gjennomsnits demand per dag:
        demand_ID = data_demand[data_demand['ID'] == ID].copy()
        demand_ID['Date'] = pd.to_datetime(demand_ID['Date'])
        demand_filtered = demand_ID[(demand_ID['Date'] >= start_dato) & (demand_ID['Date'] <= end_dato)]

        avg_demand_per_day = demand_filtered.groupby('Date')['Demand_kWh'].sum().reset_index()
        avg_demand_per_day['Avg demand kWh per day'] = avg_demand_per_day['Demand_kWh'] / 24

        avg_demand_per_day['ID'] = ID

        #Gjennomsnitss pris per dag:
        price_area = data_households[data_households['ID'] == ID].iloc[0]['Price_area']
        price_data = data_price_update[data_price_update['Price_area'] == price_area].copy()
        price_data['Date'] = pd.to_datetime(price_data['Date'])
        price_filtered = price_data[(price_data['Date'] >= start_dato) & (price_data['Date'] <= end_dato)]

        avg_price_per_day = price_filtered.groupby('Date')['Price_NOK_kWh'].sum().reset_index()
        avg_price_per_day['Avg price NOK kWh per day'] = avg_price_per_day['Price_NOK_kWh'] / 24

        # Temperatur:
        Blindern_Temperatur_dag['Date'] = pd.to_datetime(Blindern_Temperatur_dag['Date'])

        # Merge datasettene til et stort et:
        merged_1 = pd.merge(avg_demand_per_day, avg_price_per_day, on='Date')
        merged_1['ID'] = ID
        merged = pd.merge(merged_1, Blindern_Temperatur_dag, on='Date')

        filtered = merged[(merged['Avg demand kWh per day'] > 0) & (merged['Avg price NOK kWh per day'] > 0) & (
            merged['Temperatur'].notnull())].copy()

        if len(filtered) > 0:
            #Demand = beta_0 + beta_1 * log(price) + beta_2 * T^2 + beta_3 * T^3
            filtered['log_price'] = np.log(filtered['Avg price NOK kWh per day'])  # Logartitmen av strømprisen
            filtered['demand'] = filtered['Avg demand kWh per day']
            filtered['T'] = filtered['Temperatur']
            filtered['T2'] = filtered['T'] **2
            filtered['T3'] = filtered['T'] **3

            X_natural = sm.add_constant(filtered[['log_price', 'T', 'T2', 'T3']])
            y_natural = filtered['demand']
            model_natural = sm.OLS(y_natural, X_natural).fit()
            beta_natural = model_natural.params['log_price']

            regresjonslinje_natural = (f"demand = {model_natural.params['const']: .2f} + "
                                       f"{model_natural.params['log_price']: .2f} *log(price) +"
                                       f"{model_natural.params['T']: .3f} *T +"
                                       f"{model_natural.params['T2']: .4f} *T^2 +"
                                       f"{model_natural.params['T3']: .4f} *T^3")
            print('For ID: ' + str(ID))
            print(model_natural.summary())

            # Plot:
            '''plt.figure(figsize=(10, 6))
            plt.scatter(filtered['log_price'], filtered['demand'], alpha=0.5, label='Observasjonspunkt')
            plt.plot(filtered['log_price'], model_natural.predict(X_natural), color='red', label='Regresjonslinje')
            plt.xlabel('log(price)')
            plt.ylabel('demand kWh')
            plt.title(f'Prisfølsomhet for strømforbruk (lin-log) for husholdning ID {ID} per dag')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            cursor = mplcursors.cursor(
                plt.scatter(filtered['log_price'], filtered['demand'], alpha=0.5, label='Observasjonspunkt'), hover=True)
            datoer = filtered['Date'].dt.strftime('%Y-%m-%d').tolist()

            @cursor.connect("add")
            def on_add(sel):
                sel.annotation.set_text(datoer[sel.index])

            plt.show()'''

        else:
            beta_natural = np.nan
            regresjonslinje_natural = None

        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_rows', None)
        resultater.append({
            'ID': ID,
            'Prisfølsomhet (beta) for Natural logarithm': beta_natural,
            'Regresjonslinjen for lineær-log': regresjonslinje_natural
        })

    return pd.DataFrame(resultater)

def log_lin_prisfølsomhet_pris_dag(test_liste_husstander,data_demand,data_price_update,data_households, Blindern_Temperatur_dag):
    resultater = []
    start_dato = '2021-12-01'
    end_dato = '2021-12-31'

    for ID in test_liste_husstander:
        #Gjennomsnits demand per dag:
        demand_ID = data_demand[data_demand['ID'] == ID].copy()
        demand_ID['Date'] = pd.to_datetime(demand_ID['Date'])
        demand_filtered = demand_ID[(demand_ID['Date'] >= start_dato) & (demand_ID['Date'] <= end_dato)]

        avg_demand_per_day = demand_filtered.groupby('Date')['Demand_kWh'].sum().reset_index()
        avg_demand_per_day['Avg demand kWh per day'] = avg_demand_per_day['Demand_kWh'] / 24

        avg_demand_per_day['ID'] = ID

        #Gjennomsnitss pris per dag:
        price_area = data_households[data_households['ID'] == ID].iloc[0]['Price_area']
        price_data = data_price_update[data_price_update['Price_area'] == price_area].copy()
        price_data['Date'] = pd.to_datetime(price_data['Date'])
        price_filtered = price_data[(price_data['Date'] >= start_dato) & (price_data['Date'] <= end_dato)]

        avg_price_per_day = price_filtered.groupby('Date')['Price_NOK_kWh'].sum().reset_index()
        avg_price_per_day['Avg price NOK kWh per day'] = avg_price_per_day['Price_NOK_kWh'] / 24

        # Temperatur:
        Blindern_Temperatur_dag['Date'] = pd.to_datetime(Blindern_Temperatur_dag['Date'])

        # Merge datasettene til et stort et:
        merged_1 = pd.merge(avg_demand_per_day, avg_price_per_day, on='Date')
        merged_1['ID'] = ID
        merged = pd.merge(merged_1, Blindern_Temperatur_dag, on='Date')

        filtered = merged[(merged['Avg demand kWh per day'] > 0) & (merged['Avg price NOK kWh per day'] > 0) & (merged['Temperatur'].notnull())].copy()

        if len(filtered) > 0:
            # log(demand) = beta_0 + beta_1 * price
            filtered['price'] = filtered['Avg price NOK kWh per day']  # Logartitmen av strømprisen
            filtered['log_demand'] = np.log(filtered['Avg demand kWh per day'])
            filtered['T'] = filtered['Temperatur']
            filtered['T2'] = filtered['T'] **2
            filtered['T3'] = filtered['T'] **3


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
            print('For ID: ' + str(ID))
            print(model_natural.summary())

        else:
            beta_natural = np.nan
            regresjonslinje_natural = None

        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_rows', None)
        resultater.append({
            'ID': ID,
            'Prisfølsomhet (beta) for Natural logarithm': beta_natural,
            'Regresjonslinjen for log-lin': regresjonslinje_natural
        })

    return pd.DataFrame(resultater)

#-----------------------------------------------------------------------------------

'''Regresjon for "direkte", ren regresjonsanalyse: demand = beta_0 + beta_1 *pris + beta_2 *T + beta_3 *T^2 + beta_4 *T^3'''

def direkte_prisfølsomhet_dag(test_liste_husstander,data_demand,data_price_update,data_households, Blindern_Temperatur_dag):
    resultater = []
    start_dato = '2021-12-01'
    end_dato = '2021-12-31'

    for ID in test_liste_husstander:
        #Gjennomsnits demand per dag:
        demand_ID = data_demand[data_demand['ID'] == ID].copy()
        demand_ID['Date'] = pd.to_datetime(demand_ID['Date'])
        demand_filtered = demand_ID[(demand_ID['Date'] >= start_dato) & (demand_ID['Date'] <= end_dato)]

        avg_demand_per_day = demand_filtered.groupby('Date')['Demand_kWh'].sum().reset_index()
        avg_demand_per_day['Avg demand kWh per day'] = avg_demand_per_day['Demand_kWh'] / 24

        avg_demand_per_day['ID'] = ID

        #Gjennomsnitss pris per dag:
        price_area = data_households[data_households['ID'] == ID].iloc[0]['Price_area']
        price_data = data_price_update[data_price_update['Price_area'] == price_area].copy()
        price_data['Date'] = pd.to_datetime(price_data['Date'])
        price_filtered = price_data[(price_data['Date'] >= start_dato) & (price_data['Date'] <= end_dato)]

        avg_price_per_day = price_filtered.groupby('Date')['Price_NOK_kWh'].sum().reset_index()
        avg_price_per_day['Avg price NOK kWh per day'] = avg_price_per_day['Price_NOK_kWh'] / 24

        # Temperatur:
        Blindern_Temperatur_dag['Date'] = pd.to_datetime(Blindern_Temperatur_dag['Date'])

        # Merge datasettene til et stort et:
        merged_1 = pd.merge(avg_demand_per_day, avg_price_per_day, on='Date')
        merged_1['ID'] = ID
        merged = pd.merge(merged_1, Blindern_Temperatur_dag, on='Date')

        filtered = merged[(merged['Avg demand kWh per day'] > 0) & (merged['Avg price NOK kWh per day'] > 0)].copy()

        if len(filtered) > 0:
            #Regresjonslinje: Demand = beta_0 + beta_1 * pris + beta_2 * T + beta_3 *T^2 + beta_4 *T^3
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

            regresjonslinje_natural = f"demand = {model_natural.params['const']: .2f} + {model_natural.params['price']: .2f} *price + {model_natural.params['T']: .4f} *T + {model_natural.params['T2']: .4f} *T^2 + {model_natural.params['T3']: .4f} *T^3"
            print('For ID: ' + str(ID))
            print(model_natural.summary())

        else:
            beta_natural = np.nan
            regresjonslinje_natural = None

        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_rows', None)
        resultater.append({
            'ID': ID,
            'Prisfølsomhet (beta) for lineær regresjonsanalyse': beta_natural,
            'Regresjonslinjen for lineær': regresjonslinje_natural
        })

    return pd.DataFrame(resultater)


#-----------------------------------------------------------------------------------

'''Kjøre funksjonene, printer ut resultatene '''

##resultater = log_log_prisfølsomhet_dag(test_liste_husstander,data_demand, data_price_update, data_households, Blindern_Temperatur_dag)
#resultater = log_lin_prisfølsomhet_temp_dag(test_liste_husstander,data_demand, data_price_update, data_households, Blindern_Temperatur_dag)
#resultater = lin_log_prisfølsomhet_pris_dag(test_liste_husstander,data_demand,data_price_update,data_households, Blindern_Temperatur_dag)
#resultater = log_lin_prisfølsomhet_pris_dag(test_liste_husstander,data_demand,data_price_update,data_households, Blindern_Temperatur_dag)
resultater = direkte_prisfølsomhet_dag(test_liste_husstander,data_demand,data_price_update,data_households, Blindern_Temperatur_dag)

print(resultater)



