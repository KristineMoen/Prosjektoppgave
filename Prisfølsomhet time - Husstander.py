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

test_liste_husstander = [512, 642] #Bare for test

#-----------------------------------------------------------------------------------

'''Regresjon for "direkte", ren regresjonsanalyse: demand = beta_0 + beta_1 *pris + beta_2 *T + beta_3 *T^2 + beta_4 *T^3 + sum(alpha_h *time_h) + error'''

def direkte_prisfølsomhet_time(test_liste_husstander, data_demand, data_price_update, data_households, Blindern_Temp_t4t):
    resultater = []
    start_dato = '2021-12-01'
    end_dato ='2021-12-04'

    for ID in test_liste_husstander:
        # demand per time:
        demand_ID = data_demand[data_demand['ID'] == ID].copy()
        demand_ID['Date'] = pd.to_datetime(demand_ID['Date'])
        demand_filtered = demand_ID[(demand_ID['Date'] >= start_dato) & (demand_ID['Date'] <= end_dato)]


        # pris per time:
        price_area = data_households[data_households['ID'] == ID].iloc[0]['Price_area']
        price_data = data_price_update[data_price_update['Price_area'] == price_area].copy()
        price_data['Date'] = pd.to_datetime(price_data['Date'])
        price_filtered = price_data[(price_data['Date'] >= start_dato) & (price_data['Date'] <= end_dato)]

        # Temperatur:
        Blindern_Temp_t4t['Date'] = pd.to_datetime(Blindern_Temp_t4t['Date'])

        # Merge datasettene til et stort:
        merged_1 = pd.merge(demand_filtered, price_filtered, on='Date')
        merged_1['ID'] = ID
        merged = pd.merge(merged_1, Blindern_Temp_t4t, on='Date')

        filtered = merged[(merged['Demand_kWh'] > 0) & (merged['Price_NOK_kWh'] > 0) & (merged['Temperatur'].notnull())].copy()

        if len(filtered) > 0:
            filtered['demand'] = filtered['Demand_kWh']  # Logaritmen av strømforbruket
            filtered['price'] = filtered['Price_NOK_kWh']  # Logartitmen av strømprisen
            filtered['T'] = filtered['Temperatur']
            filtered['T2'] = filtered['T'] ** 2
            filtered['T3'] = filtered['T'] ** 3
            #filtered['hour'] = filtered['Hour']

            #hour_dummies = pd.get_dummies(filtered['hour'], prefix='hour')

            unike_timer = filtered['Date'].unique()

            for tidspunkt in unike_timer:
                time_data = filtered[filtered['Date'] == tidspunkt]

                # Regresjonsanalyse: demand = beta_0 + beta_1 * pris + beta_2 *T + beta_3 *T^2 + beta_4 *T^3 + sum(alpha_h *time_h) + error
                #X_natural = pd.concat([filtered[['price', 'T', 'T2', 'T3']], hour_dummies], axis = 1)
                #X_natural = sm.add_constant(X_natural)
                #X_natural = X_natural.astype(float)
                X_natural = time_data[['price', 'T', 'T2', 'T3']]
                X_natural = sm.add_constant(X_natural.astype(float))
                y_natural = filtered['demand'].astype(float)

                X_natural = X_natural.reset_index(drop = True)
                y_natural = y_natural.reset_index(drop = True)

                X_natural, y_natural = X_natural.align(y_natural, join='inner', axis=0)

                model_natural = sm.OLS(y_natural, X_natural).fit()
                beta_natural = model_natural.params['price']
                #hour_params = {param: value for param, value in model_natural.params.items() if param.startswith('hour')}
                #hour_str = " + ".join([f"{v: .2f} * {k}" for k, v in hour_params.items()])

                regresjonslinje_linear = (f"demand = {model_natural.params['const']: .2f} +"
                                      f" {model_natural.params['price']: .2f} *price + "
                                      f"{model_natural.params['T']: .2f} * T + "
                                      f"{model_natural.params['T2']: .2f} * T^2 + "
                                      f"{model_natural.params['T3']: .2f} * T^3 ")
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


def direkte_prisfølsomhet_time_for_time(test_liste_husstander, data_demand, data_price_update, data_households, Blindern_Temp_t4t):
    resultater = []
    start_dato = '2021-12-01'
    end_dato = '2021-12-06'

    for ID in test_liste_husstander:
        # --- Hent data for husholdning ---
        demand_ID = data_demand[data_demand['ID'] == ID].copy()
        demand_ID['Date'] = pd.to_datetime(demand_ID['Date'])
        demand_filtered = demand_ID[(demand_ID['Date'] >= start_dato) & (demand_ID['Date'] <= end_dato)]

        # Pris per time
        price_area = data_households[data_households['ID'] == ID].iloc[0]['Price_area']
        price_data = data_price_update[data_price_update['Price_area'] == price_area].copy()
        price_data['Date'] = pd.to_datetime(price_data['Date'])
        price_filtered = price_data[(price_data['Date'] >= start_dato) & (price_data['Date'] <= end_dato)]

        # Temperatur
        Blindern_Temp_t4t['Date'] = pd.to_datetime(Blindern_Temp_t4t['Date'])

        # Merge
        merged = pd.merge(pd.merge(demand_filtered, price_filtered, on='Date'), Blindern_Temp_t4t, on='Date')
        filtered = merged[(merged['Demand_kWh'] > 0) &
                          (merged['Price_NOK_kWh'] > 0) &
                          (merged['Temperatur'].notnull())].copy()

        if len(filtered) == 0:
            continue

        filtered['demand'] = filtered['Demand_kWh']
        filtered['price'] = filtered['Price_NOK_kWh']
        filtered['T'] = filtered['Temperatur']
        filtered['T2'] = filtered['T'] ** 2
        filtered['T3'] = filtered['T'] ** 3
        filtered['hour'] = filtered['Hour']

        print("\n================================================================")
        print(f"OLS-resultater for husholdning ID: {ID}")
        print("================================================================")

        # --- Kjør én regresjon per time ---
        for h in range(24):
            hour_data = filtered[filtered['hour'] == h]
            if len(hour_data) < 5:
                continue

            X = hour_data[['price', 'T', 'T2', 'T3']]
            X = sm.add_constant(X.astype(float))
            y = hour_data['demand'].astype(float)

            model = sm.OLS(y, X).fit()

            print("\n--------------------------------------------------------------")
            print(f"Time {h}:00  (antall observasjoner: {len(hour_data)})")
            print("--------------------------------------------------------------")
            print(model.summary())  # <== full tabell for den timen
            print("--------------------------------------------------------------")

            regresjonslinje = "demand = " + " + ".join([f"{v:.4f}*{k}" for k, v in model.params.items()])

            resultater.append({
                'ID': ID,
                'Hour': h,
                'Prisfølsomhet (beta)': model.params['price'],
                'R2': model.rsquared,
                'Antall observasjoner': len(hour_data),
                'Regresjonslinje': regresjonslinje
            })

    # --- Slutt: vis samlet tabell ---
    df_resultater = pd.DataFrame(resultater)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    print("\n================================================================")
    print("📊 FULLSTENDIG RESULTATTABELL: PRISFØLSOMHET TIME FOR TIME")
    print("================================================================\n")
    print(df_resultater)

    return df_resultater

#resultater = direkte_prisfølsomhet_time_for_time(test_liste_husstander, data_demand, data_price_update, data_households, Blindern_Temp_t4t)
resultater = direkte_prisfølsomhet_time(test_liste_husstander,data_demand,data_price_update, data_households, Blindern_Temp_t4t)

print(resultater)


