import pandas as pd
import numpy as np
from pandas import to_datetime

# --------------------- CSV - Filer -----------------------


data_demand = pd.read_csv('/Users/synnelefdal/Desktop/<3/5.klasse/demand.csv')

data_price = pd.read_csv('prices.csv')
data_price_update = data_price.drop(columns = ['Price_NOK_MWh'])

#------------------------------------- FINNE AKTUELLE HUSSTANDER -------------------------------------------#

#Finne ID:
data_answer = pd.read_csv('answers.csv')
data_households = pd.read_csv('households (1).csv')
liste_husstander = []

def finne_husstander():
    for index, rad in data_answer.iterrows():
        if (
                rad["Q_City"] in [5] and          # 5 = Bergen
                # rad["Q23"] in [1,2,3]           # 1= Under 30 kvm, 2 = 30-49 kvm, 3 = 50-59 kvm, 4 = 60-79 kvm, 5 = 80-99 kvm, 6 = 100-119 kvm, 7 = 120-159 kvm, 8 = 160-199 kvm, 9 = 200 kvm eller større, 10 = vet ikke
                # rad["Q21"] in [1,2]             # 1 = Under 300 000 kr, 2 = 300 000 - 499 999, 3 = 500 000 -799 999, 4 = 800 000 - 999 999, 5 = 1 000 000 - 1 499 999, 6 = 1 500 000 eller mer, 7 = Vil ikke oppgi, 8 = Vet ikke
                rad["Q29"] == 1 and               # 1 = Har elbil, 2 = Har ikke elbil
                rad["Q31"] in [2, 3, 4]           # 1 = Styrer ikke ladning av elbil for å unngå timer med høye priser, 2 = Ja, manuelt, 3 = Ja, automatisk etter tidspunkt, 4 = Ja, automatisk etter timepris
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


# ------------------- BEREGNING AV ALTERNATIV 1 --------------------------

def beregn_alternativ1(data_households, data_demand, data_price):

    # -------- Tak i kWh, må endres for hver sugbruppe før hver kjøring ------
    tak_kWh = 19510

    # ----- Analyse periode -----------

    start_dato = '2021-04-01'
    end_dato = '2022-03-31'

    resultater = []

    for ID in liste_husstander:

        # ------------------ Demand -------------------- #
        demand_ID = data_demand[data_demand['ID'] == ID]
        demand_ID = demand_ID.copy()
        demand_ID['Date'] = pd.to_datetime(demand_ID['Date'])

        # ------------------ Price ---------------------- #
        price_area = data_households[data_households['ID'] == ID].iloc[0]['Price_area']
        price_data = data_price[data_price['Price_area'] == price_area]
        price_data = price_data.copy()
        price_data['Date'] = pd.to_datetime(price_data['Date'])

        # ------------ Filtrering ----------------------- #
        demand_filtered = demand_ID[(demand_ID['Date'] >= start_dato) & (demand_ID['Date'] <= end_dato)]
        price_filtered = price_data[(price_data['Date'] >= start_dato) & (price_data['Date'] <= end_dato)]

        # ----------------- Merging ---------------------- #
        merged = pd.merge(demand_filtered, price_filtered, on=['Date', 'Hour'])
        merged['Price'] = merged['Demand_kWh'] * merged['Price_NOK_kWh']

        df = merged.copy()

        # -------------------- Beregninger ------------------ #

        total_price = df['Price'].sum()

        df["Date"] = pd.to_datetime(df["Date"])
        if "Hour" in df.columns:
            df['Hour'] = pd.to_numeric(df["Hour"], errors="coerce").fillna(0).astype(int)
            df["ts"] = df["Date"] + pd.to_timedelta(df["Hour"], unit="h")
        else:
            df["ts"] = df["Date"]

        df.sort_values("ts", inplace=True)

        df["Demand_kWh"] = pd.to_numeric(df["Demand_kWh"], errors="coerce").fillna(0.0)
        df['Price_NOK_kWh'] = pd.to_numeric(df["Price_NOK_kWh"], errors="coerce").fillna(0.0)

        cum_before = df["Demand_kWh"].cumsum().shift(fill_value=0)
        remaining = np.maximum(tak_kWh - cum_before, 0.0)

        df["fast_kWh"] = np.minimum(df["Demand_kWh"], remaining)
        df["spot_kWh"] = df["Demand_kWh"] - df["fast_kWh"]

        df["kostnader_fast_NOK"] = df["fast_kWh"] * 0.4
        df["kostander_spot_NOK"] = df["spot_kWh"] * df["Price_NOK_kWh"]
        df["total_kostnad_NOK"] = df["kostnader_fast_NOK"] + df["kostander_spot_NOK"]

        total_demand = df["Demand_kWh"].sum()
        total_fast_kWh = df["fast_kWh"].sum()
        total_spot_kWh = df["spot_kWh"].sum()

        # ------- Resultater ----------- #

        resultater.append({
            'Total Demand': total_demand,
            'fast_kWh': total_fast_kWh,
            'spot_kWh': total_spot_kWh,
            'NOK total alternativ 1': df["total_kostnad_NOK"].sum(),
            'Støtte': total_price - df["total_kostnad_NOK"].sum()
        })

    df_resultater = pd.DataFrame(resultater)

    # -------- Printer ut gjennomsnittet for alle husholdninger i liste over husstander ----------- #

    print("\nGjennomsnitt for alle husstander:")
    print(df_resultater.mean(numeric_only=True))


beregn_alternativ1(data_households, data_demand, data_price)
