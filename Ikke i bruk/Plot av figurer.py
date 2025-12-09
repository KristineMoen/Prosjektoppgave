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

Blindern_Temp_t4t = pd.read_csv('../Blindern_temperatur_t4t.csv')
Bergen_Temp_t4t = pd.read_csv('../Bergen_temp_t4t.csv')

#------------------------------------- FINNE AKTUELLE HUSSTANDER -------------------------------------------#

#Finne ID:
data_answer = pd.read_csv('../answers.csv')
data_households = pd.read_csv('../households (1).csv')
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

#----------------------- Plot av demand med temperatur pr dag ------------------------------

'''Funksjoner som plotter demand med temperatur og pris for de ulike ID-ene som kommer fra finne_husstander()-funksjonen'''


def plot_demand_og_temp(liste_husstander, data_demand, Blindern_Temperatur_dag):
    start_dato = '2021-04-01'
    end_dato = '2022-03-31'

    for ID in liste_husstander:
        # Gjennomsnits demand per dag:
        demand_ID = data_demand[data_demand['ID'] == ID].copy()
        demand_ID['Date'] = pd.to_datetime(demand_ID['Date'])
        demand_filtered = demand_ID[(demand_ID['Date'] >= start_dato) & (demand_ID['Date'] <= end_dato)]

        avg_demand_per_day = demand_filtered.groupby('Date')['Demand_kWh'].sum().reset_index()
        avg_demand_per_day['Avg demand kWh per day'] = avg_demand_per_day['Demand_kWh'] / 24

        avg_demand_per_day['ID'] = ID

        # Temperatur:
        Blindern_Temperatur_dag['Date'] = pd.to_datetime(Blindern_Temperatur_dag['Date'])

        merged = pd.merge(avg_demand_per_day, Blindern_Temperatur_dag, on='Date')
        merged['ID'] = ID

        filtered = merged[(merged['Avg demand kWh per day'] > 0) & (merged['Temperatur'].notnull())].copy()

        filtered['demand'] = filtered['Avg demand kWh per day']
        filtered['T'] = filtered['Temperatur']

        # Plot:
        plt.figure(figsize=(10, 6))
        plt.scatter(filtered['demand'], filtered['T'], alpha=0.5, label='Observasjonspunkt')
        plt.xlabel('Temperatur')
        plt.ylabel('demand kWh')
        plt.title(f'Plot av demand og temperatur for husholdning ID {ID} per dag i perioden {start_dato} til {end_dato}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        cursor = mplcursors.cursor(
            plt.scatter(filtered['demand'], filtered['T'], alpha=0.5, label='Observasjonspunkt'), hover=True)
        datoer = filtered['Date'].dt.strftime('%Y-%m-%d').tolist()

        @cursor.connect("add")
        def on_add(sel):
            sel.annotation.set_text(datoer[sel.index])

        plt.show()


def plot_demand_og_pris_hus(liste_husstander, data_demand, data_price_update, data_households):
    start_dato = '2021-08-01'
    end_dato = '2021-12-31'

    for ID in liste_husstander:
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
        price_filtered.loc[:, 'Price_NOK_kWh'] = price_filtered['Price_NOK_kWh'].apply(
            lambda x: x if x > 0 else 0.01)  # Dette skal fikset prisen, om den er negativ

        # Merge datasettene til et stort:
        merged = pd.merge(demand_filtered, price_filtered, on=['Date', 'Hour'])
        merged['ID'] = ID

        filtered = merged[
            (merged['Demand_kWh'] > 0) & (merged['Price_NOK_kWh'] > 0)].copy()

        print(filtered)

        # Plot:
        plt.figure(figsize=(10, 6))
        plt.scatter(filtered['Demand_kWh'], filtered['Price_NOK_kWh'], alpha=0.5, label='Observasjonspunkt')
        plt.xlabel('Price NOK')
        plt.ylabel('Demand kWh')
        plt.title(f'Plot av demand og pris for husholdning ID {ID} per dag i perioden {start_dato} til {end_dato}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        cursor = mplcursors.cursor(
            plt.scatter(filtered['Demand_kWh'], filtered['Price_NOK_kWh'], alpha=0.5, label='Observasjonspunkt'), hover=True)
        datoer = filtered['Date'].dt.strftime('%Y-%m-%d').tolist()

        @cursor.connect("add")
        def on_add(sel):
            sel.annotation.set_text(datoer[sel.index])

        plt.show()

def plot_demand_og_pris_aggregert(liste_husstander, data_demand, data_price_update, data_households):
    start_dato = '2021-08-01'
    end_dato = '2021-12-31'

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

    # Merge:
    merged = pd.merge(total_hour_demand, price_filtered, on=['Date', 'Hour'])

    filtered = merged[(merged['Demand_kWh'] > 0) & (merged['Price_NOK_kWh'] > 0)].copy()

    #print(filtered)

    # Plot:
    plt.figure(figsize=(10, 6))
    plt.scatter(filtered['Demand_kWh'], filtered['Price_NOK_kWh'], alpha=0.5, label='Observasjonspunkt')
    plt.xlabel('Price NOK')
    plt.ylabel('Demand kWh')
    plt.title(f'Plot av demand og pris for aggregert data i perioden {start_dato} til {end_dato}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    cursor = mplcursors.cursor(
        plt.scatter(filtered['Demand_kWh'], filtered['Price_NOK_kWh'], alpha=0.5, label='Observasjonspunkt'),
        hover=True)
    datoer = filtered['Date'].dt.strftime('%Y-%m-%d').tolist()

    @cursor.connect("add")
    def on_add(sel):
        sel.annotation.set_text(datoer[sel.index])

    plt.show()


#-----------------------------------------------------------------------------------

'''Kjøre funksjonene, printer ut resultatene'''

test_liste_husstander = [512] #Bare for test

#print(plot_demand_og_temp(liste_husstander, data_demand, Blindern_Temperatur_dag))
#print(plot_demand_og_pris_hus(test_liste_husstander, data_demand, data_price_update, data_households))
print(plot_demand_og_pris_aggregert(liste_husstander,data_demand,data_price_update,data_households))