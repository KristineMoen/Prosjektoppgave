#importere data
from logging.config import listen

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import row

import statsmodels.api as sm
import mplcursors

######################### LESE FILER ############################################

data_demand = pd.read_csv('/Users/kristinemoen/Documents/5-klasse/Prosjektoppgave_CSV_filer/demand.csv')
#print(data_demand)

data_price = pd.read_csv('/Users/kristinemoen/Documents/5-klasse/Prosjektoppgave_CSV_filer/prices.csv')
data_price_update = data_price.drop(columns = ['Price_NOK_MWh'])
#print(data_price)

Blindern_Temperatur_dag = pd.read_csv('Blindern_Temperatur_dag.csv')
Blindern_Temp_t4t = pd.read_csv('Blindern_temperatur_t4t.csv')


######################## FINNE AKTUELLE HUSSTANDER ##############################

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


################################### ULIKE HUSSTANDER UT I FRA ID #################################


'''ID 18 er en husholdning som bor i leilighet (80-99 kvm). De er pensjonister, har høyere utdanning og tjener 300-500k
i NO1 (OSLO), velger å hente ut data fra ID 18 og NO1: Disse hadde fastpris'''

#data_demand_ID = data_demand[data_demand['ID']==18]
#print(data_demand_ID18)

#data_price_NO1 = data_price_update[data_price['Price_area']=='NO1']
#print(data_price_NO1)


'''ID 67 er en husholdning som bor i enebolig (160-199 kvm). De jobber og har høyere utdanning + tjener 1-1.5 mill brutto
bor også i Oslo: Disse har spotpris'''

#data_demand_ID = data_demand[data_demand['ID']==67]

#data_price_NO1 = data_price_update[data_price['Price_area']=='NO1']

'''ID 67 er en husholdning som bor i enebolig (160-199 kvm). De jobber og har høyere utdanning + tjener 1.5 mill eller mer brutto
bor også i Oslo: Disse har spotpris'''

#data_demand_ID = data_demand[data_demand['ID']==67]

#data_price_NO1 = data_price_update[data_price['Price_area']=='NO1']

'''ID 512 er en husholdning som bor i enebolig (200kvm eller større). De jobber og har høyere utdanning + tjener 1-1.5 mil brutto
bor også i Oslo: Disse har spotpris'''

#data_demand_ID = data_demand[data_demand['ID']==314]

#data_price_NO1 = data_price_update[data_price['Price_area']=='NO5']



############################## TIDSROMET VI HAR VALGT ########################################

'''#Velger tidsrommet fra 2021-04-01 til 2022-03-31:
start_dato = '2021-04-01'
end_dato = '2022-03-31'

mask_demand = (data_demand_ID['Date'] >= start_dato) & (data_demand_ID['Date'] <= end_dato)
filtered_data_demand = data_demand_ID[mask_demand]
#print(filtered_data_demand)

mask_price = (data_price_NO1['Date'] >= start_dato) & (data_price_NO1['Date'] <= end_dato)
filtered_data_price = data_price_NO1[mask_price]
#print(filtered_data_price)

# Slå sammen datasett på Date og Hour
merged_data = pd.merge(filtered_data_demand, filtered_data_price, on=['Date', 'Hour'])

#Beregninger for 2021-04-01 til 2022-03-31:
merged_data['Price'] = merged_data['Demand_kWh'] * merged_data['Price_NOK_kWh']

print(merged_data[['Date', 'Hour', 'Demand_kWh', 'Price_NOK_kWh', 'Price']]'''

############### BEREGNINGER PÅ STRØM UTEN AVGIFT OG NETTLEIE MEN MED/UTEN NORGESPRIS ##############################

'''#Total pris på strøm uten avgift og nettleie fra 2021-04-01 til 2022-03-31:
total_cost = 0
for i in merged_data["Price"]:
    total_cost += i
print("Total cost:", total_cost, "i NOK. Uten noen form for strømstøtte eller Norgespris.")

#Total pris på strøm uten avgift og nettleie men med Norgespris fra 2021-04-01 til 2022-03-31:
total_cost_Norgespris = 0
for i in merged_data["Demand_kWh"]:
    total_cost_Norgespris += i*0.4
print("Total cost:", total_cost_Norgespris, "i NOK. Med Norgespris.")

diff = total_cost - total_cost_Norgespris
print(diff, "Hvis positiv tjener de på Norgespris, uten avgift og nettleie")'''

########################## PLOTTING AV STOLPEDIAGRAM ################################################


#merged_data["X_label"] = [f"d{i}, {t}" for i, t in zip(merged_data["Date"], merged_data["Hour"])]

#plt.figure(figsize=(16, 6))
#plt.bar(merged_data["X_label"], merged_data["Price"], color ="blue")
#plt.axhline(y=0.4, color ="red", linestyle = "-", label = "Norgespris")
#plt.xticks(rotation=90)
#plt.title("Forbruk time for time", fontsize=14)
#plt.xlabel("Forbruk time for time", fontsize=12)
#plt.ylabel("Kostnad i NOK", fontsize=12)
#plt.show()



############################### PRINTE OVERSIKT OVER ULIKE HUSSTANDER ########################################


def sammenlikning_av_husholdninger(data_answer, data_households, data_demand, data_price):            #list inkluderer liste over ID
    start_dato = '2021-04-01'
    end_dato = '2022-03-31'

    resultater = []


    for ID in liste_husstander:
        rad_info = data_answer[data_answer['ID'] == ID].iloc[0]
        husholdning_type = rad_info['Q22']
        størrelse = rad_info['Q23']
        by = rad_info['Q_City']
        inntekt = rad_info['Q21']
        utdanning = rad_info['Q20']
        #oppvarming = rad['']

        demand_ID = data_demand[data_demand['ID'] == ID]
        price_area = data_households[data_households['ID'] == ID].iloc[0]['Price_area']
        price_data = data_price[data_price['Price_area'] == price_area]

        # Filtrer tidsrom
        demand_ID = demand_ID.copy()
        demand_ID['Date'] = pd.to_datetime(demand_ID['Date'])
        price_data = price_data.copy()
        price_data['Date'] = pd.to_datetime(price_data['Date'])

        demand_filtered = demand_ID[(demand_ID['Date'] >= start_dato) & (demand_ID['Date'] <= end_dato)]
        price_filtered = price_data[(price_data['Date'] >= start_dato) & (price_data['Date'] <= end_dato)]

        # Merge og beregninger
        merged = pd.merge(demand_filtered, price_filtered, on=['Date', 'Hour'])
        merged['Price'] = merged['Demand_kWh'] * merged['Price_NOK_kWh']           #Regner ut hva strømmen hadde kostet uten strømstøtte og avgifter

        merged['Price_strømstøtte'] = np.where(merged['Price_NOK_kWh'] > 0.75,                             #Regner ut prisen på strømmen med dagens strømstøtte og uten avgifter
                                               merged['Demand_kWh'] * (merged['Price_NOK_kWh'] * 0.90),
                                               merged['Demand_kWh'] * merged['Price_NOK_kWh'])


        total_demand = merged['Demand_kWh'].sum()
        total_price = merged['Price'].sum()

        total_strømstøtte = merged['Price_strømstøtte'].sum()
        total_norgespris = total_demand * 0.4                                     #Regner ut hva strømmen hadde kostet med Norgespris

        diff_norgespris_og_ingen = total_price - total_norgespris
        diff_norgespris_og_strømstøtte = total_strømstøtte - total_norgespris
        diff_u_støtte_og_m_støtte = total_price - total_strømstøtte


        resultater.append({
            'ID': ID,
            'Husholdning': husholdning_type,          #Type husholdning
            'Størrelse': størrelse,                   #Sørrelsen på husholdningen
            'Utdanning': utdanning,                   #Utdanningen til de som bor der
            'By': by,                                 #Hvilken by er husstanden i
            'Inntekt': inntekt,                       #Inntekten til husstanden
            'Total demand (kWh)': total_demand,         #Total demand i kWh
            'Tot pris u/ støtte (NOK)': total_price,        #Total strømpris uten noe støtte
            'Tot pris m/ støtte (NOK)': total_strømstøtte,     #Total strømpris med støtte
            'Tot pris m/ Norgespris (NOK)': total_norgespris,    #Total pris med Norgespris som strømstøtte
            'Diff i NOK mellom Norgespris og ingen strømstøtte': diff_norgespris_og_ingen,         #Differansne mellom pris uten støtte og Norgespris
            'Diff i NOK mellom u/ støtte og m/ støtte': diff_u_støtte_og_m_støtte,                 #Differansen i uten støtte og med støtte
            'Diff i NOK mellom Norgespris og strømstøtte' : diff_norgespris_og_strømstøtte         #Differansne i pris mellom norgespris og strømstøtte
            #'Type oppvarmin' : oppvarming
        })
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        pd.set_option("display.float_format", "{:.2f}".format)
    return pd.DataFrame(resultater)



print(sammenlikning_av_husholdninger(data_answer,data_households,data_demand,data_price))


####################### REGNE PÅ PRISFØLSOMHET #########################

test_liste_husstander = [512, 642, 827]


#------------------------- Beregnelser for dag for dag, endring i de ulike dagene ---------------------------------------#

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

            regresjonslinje_log_log = f"log(demand) = {model.params['const']: .2f} + {model.params['log_price']: .2f} *log(price NOK/kWh) + {model.params['T']: .2f} *T + {model.params['T2']: .4f} *T^2 + {model.params['T3']: .4f} *T^3"

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

            #Regresjonsanalyse: log(demand) = beta_0 + beta_1 * T + error
            X_natural = sm.add_constant(filtered['T'])
            y_natural = filtered['log_demand']
            model_natural = sm.OLS(y_natural, X_natural).fit()
            beta_natural = model_natural.params['T']

            regresjonslinje_natural = f"log(demand) = {model_natural.params['const']: .2f} + {model_natural.params['T']: .2f} *T"
            print('For ID: ' + str(ID))
            print(model_natural.summary())


            # Plot:
            plt.figure(figsize=(10, 6))
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
            plt.show()

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

def lin_log_prisfølsomhet_pris_dag(test_liste_husstander, data_demand, data_price_update, data_housholds):
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

        #Merge datasettene til et stort et:
        merged = pd.merge(avg_demand_per_day, avg_price_per_day, on='Date')

        filtered = merged[(merged['Avg demand kWh per day'] > 0) & (merged['Avg price NOK kWh per day'] > 0)].copy()

        if len(filtered) > 0:
            #Demand = beta_0 + beta_1 * log(price)
            filtered['log_price'] = np.log(filtered['Avg price NOK kWh per day'])  # Logartitmen av strømprisen
            filtered['demand'] = filtered['Avg demand kWh per day']

            X_natural = sm.add_constant(filtered['log_price'])
            y_natural = filtered['demand']
            model_natural = sm.OLS(y_natural, X_natural).fit()
            beta_natural = model_natural.params['log_price']

            regresjonslinje_natural = f"demand = {model_natural.params['const']: .2f} + {model_natural.params['log_price']: .2f} *log(price)"
            print('For ID: ' + str(ID))
            print(model_natural.summary())

            # Plot:
            plt.figure(figsize=(10, 6))
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

            plt.show()

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

def log_lin_prisfølsomhet_pris_dag(test_liste_husstander,data_demand,data_price_update,data_households):
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

        #Merge datasettene til et stort et:
        merged = pd.merge(avg_demand_per_day, avg_price_per_day, on='Date')

        filtered = merged[(merged['Avg demand kWh per day'] > 0) & (merged['Avg price NOK kWh per day'] > 0)].copy()

        if len(filtered) > 0:
            # log(demand) = beta_0 + beta_1 * price
            filtered['price'] = filtered['Avg price NOK kWh per day']  # Logartitmen av strømprisen
            filtered['log_demand'] = np.log(filtered['Avg demand kWh per day'])

            X_natural = sm.add_constant(filtered['price'])
            y_natural = filtered['log_demand']
            model_natural = sm.OLS(y_natural, X_natural).fit()
            beta_natural = model_natural.params['price']

            regresjonslinje_natural = f"log(demand) = {model_natural.params['const']: .2f} + {model_natural.params['price']: .2f} *price"
            print('For ID: ' + str(ID))
            print(model_natural.summary())

            # Plot:
            plt.figure(figsize=(10, 6))
            plt.scatter(filtered['price'], filtered['log_demand'], alpha=0.5, label='Observasjonspunkt')
            plt.plot(filtered['price'], model_natural.predict(X_natural), color='red', label='Regresjonslinje')
            plt.xlabel('price')
            plt.ylabel('log(demand) kWh')
            plt.title(f'Prisfølsomhet for strømforbruk (log-lin) for husholdning ID {ID} per dag')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            cursor = mplcursors.cursor(
                plt.scatter(filtered['price'], filtered['log_demand'], alpha=0.5, label='Observasjonspunkt'),
                hover=True)
            datoer = filtered['Date'].dt.strftime('%Y-%m-%d').tolist()

            @cursor.connect("add")
            def on_add(sel):
                sel.annotation.set_text(datoer[sel.index])

            plt.show()

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

            '''# Plot:
            plt.figure(figsize=(10, 6))
            plt.scatter(filtered['price'], filtered['demand'], alpha=0.5, label='Observasjonspunkt')
            plt.plot(filtered['price'], model_natural.predict(X_natural), color='red', label='Regresjonslinje')
            plt.xlabel('price')
            plt.ylabel('demand kWh')
            plt.title(f'Prisfølsomhet for strømforbruk (direkte) for husholdning ID {ID} per dag')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            cursor = mplcursors.cursor(
                plt.scatter(filtered['price'], filtered['demand'], alpha=0.5, label='Observasjonspunkt'),
                hover=True)
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
            'Prisfølsomhet (beta) for lineær regresjonsanalyse': beta_natural,
            'Regresjonslinjen for lineær': regresjonslinje_natural
        })

    return pd.DataFrame(resultater)


#resultater = log_log_prisfølsomhet_dag(test_liste_husstander,data_demand, data_price_update, data_households, Blindern_Temperatur_dag)
#resultater = log_lin_prisfølsomhet_temp_dag(test_liste_husstander,data_demand, data_price_update, data_households, Blindern_Temperatur_dag)
#resultater = natural_log_prisfølsomhet_time(liste_husstander,data_demand,data_price_update,data_households,Blindern_Temp_t4t)
#resultater = lin_log_prisfølsomhet_pris_dag(test_liste_husstander,data_demand,data_price_update,data_households)
#resultater = log_lin_prisfølsomhet_pris_dag(test_liste_husstander,data_demand,data_price_update,data_households)
#resultater = direkte_prisfølsomhet_dag(test_liste_husstander,data_demand,data_price_update,data_households, Blindern_Temperatur_dag)

# --------------------------- Beregnelser for time for time -------------------------------------- #


def direkte_prisfølsomhet_time(test_liste_husstander, data_demand, data_price_update, data_households, Blindern_Temp_t4t):
    resultater = []
    start_dato = '2021-12-01'
    end_dato = '2021-12-31'

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
        Blindern_Temp_t4t['Hour'] = pd.to_datetime(Blindern_Temp_t4t['Hour'])

        # Merge datasettene til et stort:
        merged_1 = pd.merge(demand_filtered, price_filtered, on='Hour')
        merged_1['ID'] = ID
        merged = pd.merge(merged_1, Blindern_Temp_t4t, on='Hour')

        filtered = merged[(merged['Demand_kWh'] > 0) & (merged['Price_NOK_kWh'] > 0) & (merged['Temperatur'].notnull())].copy()

        if len(filtered) > 0:
            filtered['demand'] = filtered['Demand_kWh']  # Logaritmen av strømforbruket
            filtered['price'] = filtered['Price_NOK_kWh']  # Logartitmen av strømprisen
            filtered['T'] = filtered['Temperatur']
            filtered['T2'] = filtered['T'] ** 2
            filtered['T3'] = filtered['T'] ** 3
            filtered['hour'] = filtered['Hour'].dt.hour

            hour_dummies = pd.get_dummies(filtered['hour'], prefix = 'hour', drop_first = True)

            # Regresjonsanalyse: demand = beta_0 + beta_1 * pris + beta_2 *T + beta_3 *T^2 + beta_4 *T^3 + sum(alpha_h *time_h) + error
            X_natural = pd.concat([filtered[['price', 'T', 'T2', 'T3']], hour_dummies], axis = 1)
            X_natural = sm.add_constant(X_natural)

            X_natural = X_natural.astype(float)
            y_natural = filtered['demand'].astype(float)
            mask = X_natural.notnull().all(axis = 1) & y_natural.notnull()

            X_natural = X_natural[mask]
            y_natural = y_natural[mask]

            model_natural = sm.OLS(y_natural, X_natural).fit()
            beta_natural = model_natural.params['price']
            hour_params = {param: value for param, value in model_natural.params.items() if param.startswith('hour')}
            hour_str = " + ".join([f"{v: .2f} * {k}" for k, v in hour_params.items()])

            regresjonslinje_linear = (f"demand = {model_natural.params['const']: .2f} +"
                                      f" {model_natural.params['price']: .2f} *price + "
                                      f"{model_natural.params['T']: .2f} * T + "
                                      f"{model_natural.params['T2']: .2f} * T^2 + "
                                      f"{model_natural.params['T3']: .2f} * T^3 + "
                                      + hour_str)
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





resultater = direkte_prisfølsomhet_time(test_liste_husstander, data_demand, data_price_update, data_households, Blindern_Temp_t4t)

print(resultater)














