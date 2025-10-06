#importere data
from logging.config import listen

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

######################### LESE FILER ############################################

data_demand = pd.read_csv('/Users/kristinemoen/Documents/5-klasse/Prosjektoppgave_CSV_filer/demand.csv')
#print(data_demand)

data_price = pd.read_csv('/Users/kristinemoen/Documents/5-klasse/Prosjektoppgave_CSV_filer/prices.csv')
data_price_update = data_price.drop(columns = ['Price_NOK_MWh'])
#print(data_price)


######################## FINNE AKTUELLE HUSSTANDER ##############################

#Finne ID:
data_answer = pd.read_csv('answers.csv')
data_households = pd.read_csv('households (1).csv')
liste_hustander = []

def finne_hustander():
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
                liste_hustander.append(int(id_verdi))

    print("ID-er som oppfyller kravene:", liste_hustander)

finne_hustander()


################################### ULIKE HUSSTANDER UT I FRA ID #################################


#ID 18 er en husholdning som bor i leilighet (80-99 kvm). De er pensjonister, har høyere utdanning og tjener 300-500k
# i NO1 (OSLO), velger å hente ut data fra ID 18 og NO1: Disse hadde fastpris

#data_demand_ID = data_demand[data_demand['ID']==18]
#print(data_demand_ID18)

#data_price_NO1 = data_price_update[data_price['Price_area']=='NO1']
#print(data_price_NO1)


#ID 67 er en husholdning som bor i enebolig (160-199 kvm). De jobber og har høyere utdanning + tjener 1-1.5 mill brutto
# bor også i Oslo: Disse har spotpris

#data_demand_ID = data_demand[data_demand['ID']==67]

#data_price_NO1 = data_price_update[data_price['Price_area']=='NO1']

#ID 67 er en husholdning som bor i enebolig (160-199 kvm). De jobber og har høyere utdanning + tjener 1.5 mill eller mer brutto
# bor også i Oslo: Disse har spotpris

#data_demand_ID = data_demand[data_demand['ID']==67]

#data_price_NO1 = data_price_update[data_price['Price_area']=='NO1']

#ID 512 er en husholdning som bor i enebolig (200kvm eller større). De jobber og har høyere utdanning + tjener 1-1.5 mil brutto
# bor også i Oslo: Disse har spotpris

data_demand_ID = data_demand[data_demand['ID']==314]

data_price_NO1 = data_price_update[data_price['Price_area']=='NO5']



############################## TIDSROMET VI HAR VALGT ########################################

#Velger tidsrommet fra 2021-04-01 til 2022-03-31:
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

print(merged_data[['Date', 'Hour', 'Demand_kWh', 'Price_NOK_kWh', 'Price']])

############### BEREGNINGER PÅ STRØM UTEN AVGIFT OG NETTLEIE MEN MED/UTEN NORGESPRIS ##############################

#Total pris på strøm uten avgift og nettleie fra 2021-04-01 til 2022-03-31:
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
print(diff, "Hvis positiv tjener de på Norgespris, uten avgift og nettleie")


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


    for ID in liste_hustander:
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
        merged['Price'] = merged['Demand_kWh'] * merged['Price_NOK_kWh']

        total_demand = merged['Demand_kWh'].sum()
        total_price = merged['Price'].sum()
        total_norgespris = total_demand * 0.4
        diff = total_price - total_norgespris

        resultater.append({
            'ID': ID,
            'Type_husholdning': husholdning_type,
            'Størrelse': størrelse,
            'Utdanning': utdanning,
            'By': by,
            'Inntekt': inntekt,
            'Total_demand_kWh': total_demand,
            'Total_strømpris_NOK': total_price,
            'Total_Norgespris_NOK': total_norgespris,
            'Differanse_NOK': diff
            #'Type oppvarmin' : oppvarming
        })
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        pd.set_option("display.float_format", "{:.2f}".format)
    return pd.DataFrame(resultater)



print(sammenlikning_av_husholdninger(data_answer,data_households,data_demand,data_price))


####################### REGNE PÅ PRISFØLSOMHET #########################

import statsmodels.api as sm

filtered_pf = merged_data[(merged_data['Demand_kWh'] > 0) & (merged_data['Price_NOK_kWh'] > 0)].copy()


filtered_pf['log_demand'] = np.log(filtered_pf['Demand_kWh'])
filtered_pf['log_price'] = np.log(filtered_pf['Price_NOK_kWh'])


# Regresjonsanalyse: log(Demand) = alpha + beta * log(Price)
X = sm.add_constant(filtered_pf['log_price'])  # Legg til konstantledd
y = filtered_pf['log_demand']
model = sm.OLS(y, X).fit()


#print(model.summary())
#beta = model.params['log_price']
#print(f"Estimert prisfølsomhet (priselastisitet): {beta:.4f}")

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(filtered_pf['log_price'], filtered_pf['log_demand'], alpha=0.5, label='Observasjoner')
plt.plot(filtered_pf['log_price'], model.predict(X), color='red', label='Regresjonslinje')
plt.xlabel('log(Pris NOK/kWh)')
plt.ylabel('log(Etterspørsel kWh)')
plt.title('Prisfølsomhet for strømforbruk (log-log regresjon)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()





def beregn_prisfølsomhet_for_husholdninger(liste_hustander, data_demand, data_price_update, data_households):
    resultater = []
    start_dato = '2021-04-01'
    end_dato = '2022-03-31'

    for ID in liste_hustander:
        demand_ID = data_demand[data_demand['ID'] == ID]
        price_area = data_households[data_households['ID'] == ID].iloc[0]['Price_area']
        price_data = data_price_update[data_price_update['Price_area'] == price_area]

        demand_ID['Date'] = pd.to_datetime(demand_ID['Date'])
        price_data['Date'] = pd.to_datetime(price_data['Date'])

        demand_filtered = demand_ID[(demand_ID['Date'] >= start_dato) & (demand_ID['Date'] <= end_dato)]
        price_filtered = price_data[(price_data['Date'] >= start_dato) & (price_data['Date'] <= end_dato)]

        merged = pd.merge(demand_filtered, price_filtered, on=['Date', 'Hour'])

        filtered = merged[(merged['Demand_kWh'] > 0) & (merged['Price_NOK_kWh'] > 0)].copy()

        if len(filtered) > 10:
            filtered['log_demand'] = np.log(filtered['Demand_kWh'])
            filtered['log_price'] = np.log(filtered['Price_NOK_kWh'])

            X = sm.add_constant(filtered['log_price'])
            y = filtered['log_demand']
            model = sm.OLS(y, X).fit()

            beta = model.params['log_price']
        else:
            beta = np.nan  # Ikke nok data

        resultater.append({
            'ID': ID,
            'Prisfølsomhet (beta)': beta
        })

        # Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(filtered['log_price'], filtered['log_demand'], alpha=0.5, label='Observasjoner')
        plt.plot(filtered['log_price'], model.predict(X), color='red', label='Regresjonslinje')
        plt.xlabel('log(Pris NOK/kWh)')
        plt.ylabel('log(Etterspørsel kWh)')
        plt.title('Prisfølsomhet for strømforbruk (log-log regresjon)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return pd.DataFrame(resultater)

resultat_df = beregn_prisfølsomhet_for_husholdninger(liste_hustander, data_demand, data_price_update, data_households)
print(resultat_df)












