import numpy as np
import pandas as pd

# DETTE E FORDELING AV GRUNNSTØTTE
data_demand = pd.read_csv('/Users/synnelefdal/Desktop/<3/5.klasse/demand.csv')

data_price = pd.read_csv('prices.csv')
data_price_update = data_price.drop(columns = ['Price_NOK_MWh'])


# ------------------------------------- FINNE AKTUELLE HUSSTANDER -------------------------------------------#

# Finne ID:
data_answer = pd.read_csv('answers.csv')
data_households = pd.read_csv('households (1).csv')
liste_husstander = []


def finne_husstander():
    for index, rad in data_answer.iterrows():
        if (
                rad["Q_City"] in [1,2,4]   # 4 = Oslo, 2 = Lillestrøm, 1 = Bærum 5 = Bergen
                # rad["Q22"] == 1            # 1 = Enebolig 4 = Boligblokk
                #rad["Q23"] in  [3]         # 1= Under 30 kvm, 2 = 30-49 kvm, 3 = 50-59 kvm, 4 = 60-79 kvm, 5 = 80-99 kvm, 6 = 100-119 kvm, 7 = 120-159 kvm, 8 = 160-199 kvm, 9 = 200 kvm eller større, 10 = vet ikke
                #rad["Q21"] in [1,2]         # 1 = Under 300 000 kr, 2 = 300 000 - 499 999, 3 = 500 000 -799 999, 4 = 800 000 - 999 999, 5 = 1 000 000 - 1 499 999, 6 = 1 500 000 eller mer, 7 = Vil ikke oppgi, 8 = Vet ikke
                # rad["Q20"] == 4         # 1 = Ingen fullført utdanning, 2 = Grunnskole, 3 = Vgs, 4 = Høyskole/Uni lavere grad, 5 = Høyskol/Uni høyere grad
                # rad["Q1"] == 1          # 1 = Fulgte med på egen strømbruk, 2 = følgte ikke med
                # rad['Q4'] == 4         # 1 = Fulgte med hver dag, 2 = Fulgte med noen ganger i uken, 3 = Fulgte med noen ganger i mnd, 4 = Fulgte med noen ganger i løpet av vinteren
                # rad["Q29"] == 1        # 1 = Ja, 2 = Nei
                # rad["Q8_12"] == 0      # 0 = Flyttet ikke elbilladning til andre timer, 1 = flyttet elbilladning til andre timer
                # rad["Q7"] == 3         # 1 = Gjorde ofte tiltak, 2 = Gjorde av og til tiltak, 3 = Nei
                # rad["Q29"] == 1   and     # 1 = Har elbil, 2 = Har ikke elbil
                # rad["Q8_13"] == 1      # 0 = Installerte ikke elbillader, 1 = Installerte elbillader
                #rad["Q31"] in [2,3,4]        # 1 = Styrer ikke ladning av elbil for å unngå timer med høye priser, 2 = Ja, manuelt, 3 = Ja, automatisk etter tidspunkt, 4 = Ja, automatisk etter timepris

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

# ---------------------------------- PRINTE OVERSIKT OVER ULIKE HUSSTANDER -------------------------------- #


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

        #Regne ut og anta
        P_akseptabelpris = 0.4 #NOK/kWh
        E_egenandel_gjennomsnitt = P_akseptabelpris *   11317.676617  #dette e tatt fra kjøring av bare bergen household gjennomsnitt total demand
        #E_egenandel = P_akseptabelpris * total_demand
        #print('Utgift strøm',E_egenandel)
        K =    12814.362665 #dette e og tatt fra bergen kjøring. tot pris uten støtte
        støtte_p_pers= K - E_egenandel_gjennomsnitt
        #print('Støtte p pers', støtte_p_pers)



        #norgespris osv
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
            'Diff i NOK mellom Norgespris og strømstøtte' : diff_norgespris_og_strømstøtte,         #Differansne i pris mellom norgespris og strømstøtte
            #'Type oppvarmin' : oppvarming
            'Støtte alternativ 2' : støtte_p_pers,
            'Strøm utgift alternativ 2' : total_price - støtte_p_pers
        })
        #pd.set_option("display.max_rows", None)
        #pd.set_option("display.max_columns", None)
        #pd.set_option("display.width", 1000)
        #pd.set_option("display.float_format", "{:.2f}".format)

    df = pd.DataFrame(resultater)

    # Print hele tabellen
    print(df)

    # Print gjennomsnitt av alle numeriske kolonner
    print("\nGjennomsnitt for alle husstander:")
    print(df.mean(numeric_only=True))

    return df


    #return pd.DataFrame(resultater)



print(sammenlikning_av_husholdninger(data_answer,data_households,data_demand,data_price))





