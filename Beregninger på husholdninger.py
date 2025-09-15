#importere data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv


data_demand = pd.read_csv('/Users/kristinemoen/Documents/5-klasse/Prosjektoppgave_CSV_filer/demand.csv')
#print(data_demand)

data_price = pd.read_csv('/Users/kristinemoen/Documents/5-klasse/Prosjektoppgave_CSV_filer/prices.csv')
data_price_update = data_price.drop(columns = ['Price_NOK_MWh'])
#print(data_price)


#ID 18 er en husholdning i NO1, velger å hente ut data fra ID 18 og NO1:
data_demand_ID18 = data_demand[data_demand['ID']==18]
#print(data_demand_ID18)

data_price_NO1 = data_price_update[data_price['Price_area']=='NO1']
#print(data_price_NO1)

#Velger tidsrommet fra 2020-10-01 til 2021-09-30:
start_dato = '2020-10-01'
end_dato = '2021-09-30'

mask_demand = (data_demand_ID18['Date'] >= start_dato) & (data_demand_ID18['Date'] <= end_dato)
filtered_data_demand = data_demand_ID18[mask_demand]
#print(filtered_data_demand)

mask_price = (data_price_NO1['Date'] >= start_dato) & (data_price_NO1['Date'] <= end_dato)
filtered_data_price = data_price_NO1[mask_price]
#print(filtered_data_price)

# Slå sammen datasett på Date og Hour
merged_data = pd.merge(filtered_data_demand, filtered_data_price, on=['Date', 'Hour'])

#Beregninger for 2020-10-01 til 2021-09-30:
merged_data['Price'] = merged_data['Demand_kWh'] * merged_data['Price_NOK_kWh']

print(merged_data[['Date', 'Hour', 'Demand_kWh', 'Price_NOK_kWh', 'Price']])


#Stoltediagram:

antall_dager = len(merged_data) // 24
merged_data["Hour"] = [i % 24 + 1 for i in range(len(merged_data))]
merged_data["Date"] = [i // 24 + 1 for i in range(len(merged_data))]

merged_data["X_label"] = [f"d{i}, {t}" for i, t in zip(merged_data["Date"], merged_data["Hour"])]

plt.figure(figsize=(16, 6))
plt.bar(merged_data["X_label"], merged_data["Price"], color ="blue")
plt.axhline(y=0.4, color ="red", linestyle = "-", label = "Norgespris")
plt.xticks(rotation=90)
plt.title("Forbruk time for time", fontsize=14)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Kostnad i NOK", fontsize=12)
plt.show()












