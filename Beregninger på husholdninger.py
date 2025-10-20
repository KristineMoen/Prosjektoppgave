#importere data
from logging.config import listen

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

import row

######################### LESE FILER ############################################

data_demand = pd.read_csv('/Users/kristinemoen/Documents/5-klasse/Prosjektoppgave_CSV_filer/demand.csv')
#print(data_demand)

data_price = pd.read_csv('/Users/kristinemoen/Documents/5-klasse/Prosjektoppgave_CSV_filer/prices.csv')
data_price_update = data_price.drop(columns = ['Price_NOK_MWh'])
#print(data_price)

Blindern_Temperatur = pd.read_csv('Blindern_Temperatur.csv')


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

        # Merge og beregninger:
        merged = pd.merge(demand_filtered, price_filtered, on=['Date', 'Hour'])
        merged['Price'] = merged['Demand_kWh'] * merged['Price_NOK_kWh']           #Regner ut hva strømmen hadde kostet uten strømstøtte

        # Regner ut prisen på strømmen med dagens strømstøtte:
        merged['Price_strømstøtte'] = np.where(merged['Price_NOK_kWh'] > 0.75,
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

import statsmodels.api as sm
from LinearRegression import LinearRegression


########## Første forsøk, dette printer hele datasettet ################
''' filtered_pf = merged_data[(merged_data['Demand_kWh'] > 0) & (merged_data['Price_NOK_kWh'] > 0)].copy()

filtered_pf['log_demand'] = np.log(filtered_pf['Demand_kWh'])
filtered_pf['log_price'] = np.log(filtered_pf['Price_NOK_kWh'])


# Regresjonsanalyse: log(Demand) = alpha + beta * log(Price)
X = sm.add_constant(filtered_pf['log_price'])  # Legg til konstantledd
y = filtered_pf['log_demand']
model = sm.OLS(y, X).fit()


print(model.summary())
beta = model.params['log_price']
print(f"Estimert prisfølsomhet (priselastisitet): {beta:.4f}")

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
plt.show() '''





def beregn_prisfølsomhet_for_husholdninger_per_dag(liste_husstander, data_demand, data_price_update, data_households, Blindern_Temperatur):    # log-log logaritme:
    resultater = []
    start_dato = '2021-04-01'
    end_dato = '2022-03-31'

    avg_demand_per_day =[]
    avg_price_per_day = []
    alle_husholdninger = []

    for ID in liste_husstander:
        #gjennomsnits demand per dag:
        demand_ID = data_demand[data_demand['ID'] == ID].copy()
        demand_ID['Date'] = pd.to_datetime(demand_ID['Date'])
        demand_filtered = demand_ID[(demand_ID['Date'] >= start_dato) & (demand_ID['Date'] <= end_dato)]

        avg_demand_per_day = demand_filtered.groupby('Date')['Demand_kWh'].sum().reset_index()
        avg_demand_per_day['Avg demand kWh per day'] = avg_demand_per_day['Demand_kWh']/24

        avg_demand_per_day['ID'] = ID
        #avg_demand_per_day.append(avg_demand_per_day)

        #gjennomsnitss pris per dag:
        price_area = data_households[data_households['ID'] == ID].iloc[0]['Price_area']
        price_data = data_price_update[data_price_update['Price_area'] == price_area].copy()
        price_data['Date'] = pd.to_datetime(price_data['Date'])
        price_filtered = price_data[(price_data['Date'] >= start_dato) & (price_data['Date'] <= end_dato)]

        avg_price_per_day = price_filtered.groupby('Date')['Price_NOK_kWh'].sum().reset_index()
        avg_price_per_day['Avg price NOK kWh per day'] = avg_price_per_day['Price_NOK_kWh']/24

        #Temperatur:
        Blindern_Temperatur['Date'] = pd.to_datetime(Blindern_Temperatur['Date'])

        #Merge datasettene til et stort et:
        merged_1 = pd.merge(avg_demand_per_day, avg_price_per_day, on = 'Date')
        merged_1['ID'] = ID
        merged = pd.merge(merged_1, Blindern_Temperatur, on = 'Date')
        #alle_husholdninger.append(merged)
        #alle_husholdninger_df = pd.concat(alle_husholdninger, ignore_index = True)


        filtered = merged[(merged['Avg demand kWh per day'] > 0) & (merged['Avg price NOK kWh per day'] > 0) & (merged['Temperatur'].notnull())].copy()

        if len(filtered) > 10:
            filtered['log_demand'] = np.log(filtered['Avg demand kWh per day'])                  # Logartitmen av strømprisen
            filtered['log_price'] = np.log(filtered['Avg price NOK kWh per day'])                # Logaritmen av strømforbruket
            filtered['T'] = filtered['Temperatur']
            filtered['T2'] =  filtered['T'] ** 2
            filtered['T3'] =  filtered['T'] ** 3



            # Regresjonsanalyse: log(Demand) = beta_0 + beta_1 * log(Price) + beta_2 * T_i + beta_3 *T_i^2 + beta_4 *T_i^3 + error
            X = filtered[['log_price', 'T', 'T2', 'T3']]
            X = sm.add_constant(X)             # Legger til en konstant
            y = filtered['log_demand']                             # Setter opp responsvariabelen
            model = sm.OLS(y, X).fit()                        # Kjører en lineær regresjonsanalyse, finner den beste linjen som passer dataene
            beta_log_log = model.params['log_price']                       # Forteller om prisfølsomheten

            regresjonslinje_log_log = f"log(demand) = {model.params['const']: .2f} + {model.params['log_price']: .2f} *log(price NOK/kWh) + {model.params['T']: .2f} + {model.params['T2']: .2f} + {model.params['T3']: .2f}"

            print('For ID: ' + str(ID))
            print(model.summary())

            T_mean = filtered['T'].sum()/len(filtered['T'])
            T2_mean = T_mean * T_mean
            T3_mean = T_mean * T_mean * T_mean

            x_vals = np.linspace(filtered['log_price'].min(), filtered['log_price'].max(), 100)
            X_plot = pd.DataFrame({
                'const': 1,
                'log_price': x_vals,
                'T': T_mean,
                'T2': T2_mean,
                'T3': T3_mean
            })
            y_pred = model.predict(X_plot)

            # Plot
            plt.figure(figsize=(10, 6))
            plt.scatter(filtered['log_price'], filtered['log_demand'], alpha=0.5, label='Observasjoner')
            plt.plot(x_vals, y_pred, color='red', linewidth=2, label=f'Regresjonslinje (T={T_mean:.1f}°C)')
            plt.xlabel('log(Price NOK/kWh)')
            plt.ylabel('log(Demand in kWh)')
            plt.title(f'Prisfølsomhet for strømforbruk (log-log regresjon) for husholdning ID {ID} per dag')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()



            #natural logarithm:


            #Regresjonsanalyse: log(demand) = beta_0 + beta_1 * T + error
            X = sm.add_constant(filtered['T'])
            y = filtered['log_demand']
            model = sm.OLS(y,X).fit()
            beta_natural = model.params['T']

            regresjonslinje_natural = f"log(demand) = {model.params['const']: .2f} + {model.params['T']: .2f} *beta_1"
            print('For ID: ' + str(ID))
            print(model.summary())


            #Plot:
            plt.figure(figsize=(10,6))
            plt.scatter(filtered['T'],filtered['log_demand'], alpha = 0.5, label = 'Observasjonspunkt')
            plt.plot(filtered['T'], model.predict(X), color = 'red', label = 'Regresjonslinje')
            plt.xlabel('Temperatur')
            plt.ylabel('log(demand kWh)')
            plt.title(f'Prisfølsomhet for strømforbruk (natural logarithm) for husholdning ID {ID} per dag')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        else:
            beta = np.nan  # Ikke nok data

        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_rows', None)
        resultater.append({
            'ID': ID,
            'Prisfølsomhet (beta) for log log logarithm': beta_log_log,
            'Regresjonsliste for log log logarithm': regresjonslinje_log_log,
            'Prisfølsomhet (beta) for Naturallogarithm': beta_natural,
            'Regresjonslinjen for Natural logarithm': regresjonslinje_natural
        })


    return pd.DataFrame(resultater)

resultater = beregn_prisfølsomhet_for_husholdninger_per_dag(liste_husstander, data_demand, data_price_update, data_households, Blindern_Temperatur)
print(resultater)


'''def avg_demand_per_day(liste_husstander, data_demand):        # Summerer opp demand for en og en ID per dag og legger det til i total_demand_list
    start_dato = '2021-04-01'
    end_dato = '2022-03-31'

    total_demand_list = []

    for ID in liste_husstander:

        demand_ID = data_demand[data_demand['ID'] == ID].copy()
        demand_ID['Date'] = pd.to_datetime(demand_ID['Date'])

        demand_filtered = demand_ID[(demand_ID['Date'] >= start_dato) & (demand_ID['Date'] <= end_dato)]

        total_demand_per_dag = demand_filtered.groupby('Date')['Demand_kWh'].sum().reset_index()
        total_demand_per_dag['Gjennomsnitt demand kWh per dag'] = total_demand_per_dag['Demand_kWh']/24

        total_demand_per_dag['ID'] = ID
        total_demand_list.append(total_demand_per_dag)


    total_demand_list = pd.concat(total_demand_list, ignore_index= True)         #Lage en dataframe

    total_demand_list = total_demand_list[['ID', 'Date', 'Demand_kWh', 'Gjennomsnitt demand kWh per dag']]


    return total_demand_list
#print(avg_demand_per_day(liste_hustander,data_demand))



def temp_korrigering(liste_husstander, Blindern_Temperatur):
    start_dato = '2021-04-01'
    end_dato = '2022-03-31'

    Blindern_Temperatur['Tid(norsk normaltid)'] = pd.to_datetime(Blindern_Temperatur['Tid(norsk normaltid)'])
    Blindern_Temperatur_filtered = Blindern_Temperatur[(Blindern_Temperatur['Tid(norsk normaltid)'] >= start_dato) & (Blindern_Temperatur['Tid(norsk normaltid)'] <= end_dato)]

    total_demand_list = avg_demand_per_day(liste_husstander, data_demand)

    temp_ref = 17  #Setter referanse temperatur til 17 grader
    HDD = 0       #Heating Degree Days, HDD

    temp_korr_liste = []

    for ID in liste_husstander:
        for index, row in Blindern_Temperatur_filtered.iterrows():
            HDD = max(0, temp_ref - row['Middeltemperatur (døgn)'])
            temp_korr_liste.append(HDD)


    temp_korr_df = pd.DataFrame({'Temperatur korrigering': temp_korr_liste})
    total_demand_list_new = pd.concat([total_demand_list, temp_korr_df], axis = 1)  # Lage en dataframe
    total_demand_list_new = total_demand_list_new[['ID', 'Date', 'Demand_kWh', 'Gjennomsnitt demand kWh per dag', 'Temperatur korrigering']]

    return total_demand_list_new
#print(temp_korrigering(liste_husstander, Blindern_Temperatur))


#Videre arbeid: finne ut hva de ulike betaene står for, sette opp regresjonsnalaysen, plotte resultate


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def regresjonsanalyse_temp(liste_husstander, data_demand):
    total_demand_list_new = temp_korrigering(liste_husstander, Blindern_Temperatur)

    resultater = []


    for ID in liste_husstander:
        #demand_ID = data_demand[data_demand['ID'] == ID]
        #total_demand = total_demand_list_new[total_demand_list_new['ID'] == ID].iloc[0]['Gjennomsnitt demand kWh per dag']
        #temp_korr = total_demand_list_new[total_demand_list_new['ID'] == ID].iloc[0]['Temperatur korrigering']

        #x = total_demand_list_new[['Temperatur korrigering']]
        #y = total_demand_list_new['Gjennomsnitt demand kWh per dag']

        df_ID = total_demand_list_new[total_demand_list_new['ID'] == ID]

        x = df_ID[['Temperatur korrigering']]
        y = df_ID['Gjennomsnitt demand kWh per dag']

        modell = LinearRegression()
        modell.fit(x,y)



        koeff = modell.coef_[0]
        konstantledd = modell.intercept_
        r2 = r2_score(y, modell.predict(x))

        # Plot: Plotter demand og temperarur korrigeringen, hvordan forbruket endrer seg med temperaturen
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, alpha=0.5, label='Observasjoner')
        plt.plot(x, modell.predict(x), color='red', label='Regresjonslinje')
        plt.xlabel('Temperatur korrigert')
        plt.ylabel('Demand kWh')
        plt.title('Prisfølsomhet for strømforbruk korrigert for ute-temp' + str(ID))
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


        resultater.append({
            'ID': ID,
            'Koeffisinet': float(koeff),
            'Konstantledd': float(konstantledd),
            'r2_score': float(r2)

        })


    return pd.DataFrame(resultater)



print(regresjonsanalyse_temp(liste_husstander, data_demand))'''


















