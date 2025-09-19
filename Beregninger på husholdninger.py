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
liste_hustander = []

def finne_hustander():
    for index, rad in data_answer.iterrows():
        if (
                rad["Q_City"] == 5 and  # 4 = Oslo
                rad["Q22"] == 1 and  # 1 = enebolig
                rad["Q23"] == 9 and  # 9 = 200kvm eller større
                rad["Q21"] == 6  # 5 = 1 - 1.5 mill        6 = 1.5 mill eller mer
        ):
            liste_hustander.append(int(rad["ID"]))  # Legger til de ulike ID-ene
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

data_demand_ID = data_demand[data_demand['ID']==512]

data_price_NO1 = data_price_update[data_price['Price_area']=='NO1']






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

#Stoltediagram:

#merged_data["X_label"] = [f"d{i}, {t}" for i, t in zip(merged_data["Date"], merged_data["Hour"])]




#plt.figure(figsize=(16, 6))
#plt.bar(merged_data["X_label"], merged_data["Price"], color ="blue")
#plt.axhline(y=0.4, color ="red", linestyle = "-", label = "Norgespris")
#plt.xticks(rotation=90)
#plt.title("Forbruk time for time", fontsize=14)
#plt.xlabel("Forbruk time for time", fontsize=12)
#plt.ylabel("Kostnad i NOK", fontsize=12)
#plt.show()












