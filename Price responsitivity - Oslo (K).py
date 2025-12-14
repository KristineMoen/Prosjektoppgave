import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import csv
import row

import statsmodels.api as sm
import patsy
import seaborn as sns
from pandas import to_datetime

# -------------------------------------- LESER DATA --------------------------------------#

data_demand = pd.read_csv('/Users/kristinemoen/Documents/5-klasse/Prosjektoppgave_CSV_filer/demand.csv')

data_price = pd.read_csv('prices.csv')
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
                rad["Q_City"] in [4, 1, 2] and    # 4 = Oslo, 2 = Lillestrøm, 1 = Bærum
                #rad["Q23"] in [1,2,3]         # 1= Under 30 kvm, 2 = 30-49 kvm, 3 = 50-59 kvm, 4 = 60-79 kvm, 5 = 80-99 kvm, 6 = 100-119 kvm, 7 = 120-159 kvm, 8 = 160-199 kvm, 9 = 200 kvm eller større, 10 = vet ikke
                rad["Q21"] in [5,6]          # 1 = Under 300 000 kr, 2 = 300 000 - 499 999, 3 = 500 000 -799 999, 4 = 800 000 - 999 999, 5 = 1 000 000 - 1 499 999, 6 = 1 500 000 eller mer, 7 = Vil ikke oppgi, 8 = Vet ikke
                #rad["Q29"] == 1   and      # 1 = Har elbil, 2 = Har ikke elbil
                #rad["Q31"] in [2,3,4]          # 1 = Styrer ikke ladning av elbil for å unngå timer med høye priser, 2 = Ja, manuelt, 3 = Ja, automatisk etter tidspunkt, 4 = Ja, automatisk etter timepris
        ):

            # Sjekk om ID finnes i data_households og har Demand_data = 'Yes'
            id_verdi = rad["ID"]
            match = data_households[
                (data_households["ID"] == id_verdi) &
                (data_households["Demand_data"] == "Yes")
                ]
            if not match.empty:
                liste_husstander.append(int(id_verdi))

    print("ID-er som oppfyller kravene:", len(liste_husstander))

finne_husstander()


def price_responsitivity(liste_husstander, data_demand, data_price_update, data_households, Blindern_Temp_t4t):
    # Definer perioder
    ref_start = '2019-07-01'  # Start for referanseperiode
    start_dato = '2021-09-01'
    end_dato = '2022-03-31'

    # ------------------ Demand -------------------- #
    data_demand['Date'] = pd.to_datetime(data_demand['Date'])
    data_demand['Hour'] = data_demand['Hour'].astype(int)

    demand_data_filtered = data_demand[(data_demand['ID'].isin(liste_husstander)) &
                                       (data_demand['Date'] >= ref_start) &
                                       (data_demand['Date'] <= end_dato)].copy()

    total_hour_demand = demand_data_filtered.groupby(['Date', 'Hour'])['Demand_kWh'].sum().reset_index()

    # ----------------------- Pris -------------------- #
    price_area = data_households[data_households['ID'].isin(liste_husstander)].iloc[0]['Price_area']
    price_data = data_price_update[data_price_update['Price_area'] == price_area].copy()
    price_data['Date'] = pd.to_datetime(price_data['Date'])
    price_data['Hour'] = price_data['Hour'].astype(int)

    price_filtered = price_data[(price_data['Date'] >= ref_start) & (price_data['Date'] <= end_dato)]
    price_filtered = price_filtered.copy()
    price_filtered['Price_NOK_kWh'] = price_filtered['Price_NOK_kWh'].apply(lambda x: x if x > 0 else 0.01)

    # ------------ Temperatur --------------------- #
    Blindern_Temp_t4t['Date'] = pd.to_datetime(Blindern_Temp_t4t['Date'])
    Blindern_Temp_t4t['Hour'] = Blindern_Temp_t4t['Hour'].astype(int)
    Blindern_Temp_t4t['Temperatur24'] = Blindern_Temp_t4t['Temperatur'].rolling(window=24, min_periods=1).mean()
    Blindern_Temp_t4t['Temperatur72'] = Blindern_Temp_t4t['Temperatur'].rolling(window=72, min_periods=1).mean()

    temp_filtered = Blindern_Temp_t4t[(Blindern_Temp_t4t['Date'] >= ref_start) &
                                      (Blindern_Temp_t4t['Date'] <= end_dato)]

    # ------------------- Merge data ---------------- #
    merged_1 = pd.merge(total_hour_demand, price_filtered, on=['Date', 'Hour'])
    merged = pd.merge(merged_1, temp_filtered, on=['Date', 'Hour'])

    filtered = merged[(merged['Demand_kWh'] > 0) & (merged['Price_NOK_kWh'] > 0) &
                      (merged['Temperatur'].notnull())].copy()

    df = pd.DataFrame(filtered)

    # ---------------- Beregeninger ---------------- #

    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.strftime('%B')

    df['Hour'] = pd.Categorical(df['Hour'].astype(str),
        categories=[str(i) for i in range(1, 25)], ordered=True)
    df['Month'] = pd.Categorical(df['Month'],
        categories=['January', 'February', 'March', 'April', 'May', 'June',
                    'July', 'August', 'September', 'October', 'November', 'December'],
        ordered=True)

    df['Price_Group'] = np.where(df['Date'] < pd.to_datetime(start_dato), 'Before_ref', pd.cut(df['Price_NOK_kWh'],
               bins=[0, 0.12, 0.55, 1.77, 6.54],
               labels=['Low', 'Medium', 'High', 'Very High'],
               include_lowest=True)
    )

    df['Price_Group'] = pd.Categorical(df['Price_Group'],
                                       categories=['Before_ref', 'Low', 'Medium', 'High', 'Very High'],
                                       ordered = True)

    print(df['Price_Group'].value_counts())


    # --- Regresjons analyse ---
    y, X = patsy.dmatrices('Demand_kWh ~ C(Price_Group, Treatment(reference="Before_ref")) + Temperatur24 + '
                           'I(Temperatur24**2) + I(Temperatur24**3) + Temperatur72 + '
                           'C(Hour, Treatment(reference="1")) + C(Month, Treatment(reference="September"))',
                           data=df, return_type='dataframe', NA_action='drop')

    model = sm.OLS(y, X).fit()
    print(model.summary())

    # ----------- PLOT AV RESULTATER ----------------- #
    # -------- Boksplott over priskategoriene --------- #

    '''sns.set(style="whitegrid")

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Price_Group', y='Demand_kWh', data=df, hue='Price_Group', palette='Set2', legend=False)
    plt.title('Fordeling av etterspørsel (kWh) per priskategori', fontsize=14)
    plt.xlabel('Priskategori', fontsize=12)
    plt.ylabel('Etterspørsel (kWh)', fontsize=12)
    plt.tight_layout()
    plt.show()

    # ----------- Plott over hvordan etterspørselen endres med temperatur --------- #

    beta_2 = model.params['Temperatur24']
    beta_3 = model.params['I(Temperatur24 ** 2)']
    beta_4 = model.params['I(Temperatur24 ** 3)']

    temp_range = np.linspace(-20, 30, 200)
    temp_effect = beta_2 * temp_range + beta_3 * temp_range ** 2 + beta_4 * temp_range ** 3

    plt.figure(figsize=(10, 6))
    plt.plot(temp_range, temp_effect, color='green', linewidth=2)
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Temperatur (°C)')
    plt.ylabel('Kalkulert effekt (kWh)')
    plt.title('Effekt av Temperatur24 basert på beta-ene')
    plt.grid(True)
    plt.show()

    # ---------- PLott over effekten av timervariabelen (Hour), viser hvordan etterspørselen endres gjennom døgnet relativt til time 1 --------- #

    hours = list(range(1, 25))

    hour_1 = 0
    hour_2 = model.params['C(Hour, Treatment(reference="1"))[T.2]']
    hour_3 = model.params['C(Hour, Treatment(reference="1"))[T.3]']
    hour_4 = model.params['C(Hour, Treatment(reference="1"))[T.4]']
    hour_5 = model.params['C(Hour, Treatment(reference="1"))[T.5]']
    hour_6 = model.params['C(Hour, Treatment(reference="1"))[T.6]']
    hour_7 = model.params['C(Hour, Treatment(reference="1"))[T.7]']
    hour_8 = model.params['C(Hour, Treatment(reference="1"))[T.8]']
    hour_9 = model.params['C(Hour, Treatment(reference="1"))[T.9]']
    hour_10 = model.params['C(Hour, Treatment(reference="1"))[T.10]']
    hour_11 = model.params['C(Hour, Treatment(reference="1"))[T.11]']
    hour_12 = model.params['C(Hour, Treatment(reference="1"))[T.12]']
    hour_13 = model.params['C(Hour, Treatment(reference="1"))[T.13]']
    hour_14 = model.params['C(Hour, Treatment(reference="1"))[T.14]']
    hour_15 = model.params['C(Hour, Treatment(reference="1"))[T.15]']
    hour_16 = model.params['C(Hour, Treatment(reference="1"))[T.16]']
    hour_17 = model.params['C(Hour, Treatment(reference="1"))[T.17]']
    hour_16 = model.params['C(Hour, Treatment(reference="1"))[T.16]']
    hour_18 = model.params['C(Hour, Treatment(reference="1"))[T.18]']
    hour_19 = model.params['C(Hour, Treatment(reference="1"))[T.19]']
    hour_20 = model.params['C(Hour, Treatment(reference="1"))[T.20]']
    hour_21 = model.params['C(Hour, Treatment(reference="1"))[T.21]']
    hour_22 = model.params['C(Hour, Treatment(reference="1"))[T.22]']
    hour_23 = model.params['C(Hour, Treatment(reference="1"))[T.23]']
    hour_24 = model.params['C(Hour, Treatment(reference="1"))[T.24]']
    hour_list = [hour_1, float(hour_2), float(hour_3), float(hour_4), float(hour_5), float(hour_6), float(hour_7),
                 float(hour_8), float(hour_9), float(hour_10),
                 float(hour_11), float(hour_12), float(hour_13), float(hour_14), float(hour_15), float(hour_16),
                 float(hour_17), float(hour_18), float(hour_19), float(hour_20),
                 float(hour_21), float(hour_22), float(hour_23), float(hour_24)]

    plt.figure(figsize=(10, 6))
    plt.plot(hours, hour_list, color='green', linewidth=2)
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Timer')
    plt.ylabel('Beta-verdier for hver time')
    plt.title('Betaene til hver time mot timer i døgnet')
    plt.grid(True)
    plt.show()

    # ------ Plott over gjennomsnittet forbruk over de ulike timene i døgnet ------------ #


    avg_hour_demand = total_hour_demand.groupby('Hour')['Demand_kWh'].mean()
    intercept = model.params['Intercept']

    hour_list_1 = [intercept]
    for h in range(2, 25):
        param_name = f'C(Hour, Treatment(reference= \"1\"))[T.{h}]'
        if param_name in model.params:
            hour_list_1.append(intercept + model.params[param_name])
        else:
            hour_list_1.append(intercept)

    hours_3 = list(range(1, 25))

    plt.figure(figsize=(10, 6))
    plt.plot(hours_3, avg_hour_demand, color='green', linewidth=2, label='Gjennomsnitt')
    plt.plot(hours_3, hour_list_1, color='blue', linewidth=2, linestyle='--', label='Modellpredikasjon')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Timer')
    plt.ylabel('Etterspørsel (kWh)')
    plt.title('Gjennomsnitt vs modellpredikert forbruk per time')
    plt.grid(True)
    plt.legend()
    plt.show()'''

    # -------- Printe sammenlignbare versjoner av lineær ----------
    print('Low', model.params['C(Price_Group, Treatment(reference="Before_ref"))[T.Low]'] / len(liste_husstander))

    print('Medium', model.params['C(Price_Group, Treatment(reference="Before_ref"))[T.Medium]'] / len(liste_husstander))

    print('High', model.params['C(Price_Group, Treatment(reference="Before_ref"))[T.High]'] / len(liste_husstander))

    print('Very High',
          model.params['C(Price_Group, Treatment(reference="Before_ref"))[T.Very High]'] / len(liste_husstander))


print(price_responsitivity(liste_husstander, data_demand, data_price_update, data_households, Blindern_Temp_t4t))
