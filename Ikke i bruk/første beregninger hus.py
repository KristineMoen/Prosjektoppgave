#importere data
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel(r'/Users/kristinemoen/Desktop/Exp_56.xlsx') #Denne filen er Exp_56, et rekkehus i Oslo i Phase 1
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
print(data)

#her skal vi prøve å beregne et hus (rekkehus i første omgang, i Oslo)
#vi velger oss et random hus som testobjekt

#UTEN Norgepris:

data["Cost per Hour"] = data["Demand_kWh"] * data["Price/kWh"]
print(data)

# MED Norgespris:

total_cost_Norgespris = 0
for i in range(len(data["Demand_kWh"])):
    total_cost_Norgespris += (i * 40)

print("Total cost med Norgespris:", total_cost_Norgespris, "øre i NOK, og i kroner blir det:", total_cost_Norgespris // 100 )


total_cost = 0
for i in range(len(data["Cost per Hour"])):
    total_cost += i
print("Total cost:", total_cost, "i NOK. Uten noen form for strømstøtte eller Norgespris. Og i kroner blir der:", total_cost // 100)

#Stoltediagram
antall_dager = len(data) // 24
data["Time"] = [i % 24 + 1 for i in range(len(data))]
data["Dag"] = [i // 24 + 1 for i in range(len(data))]

data["X_label"] = [f"d{i}, {t}" for i, t in zip(data["Dag"], data["Time"])]

plt.figure(figsize=(16, 6))
plt.bar(data["X_label"], data["Cost per Hour"], color ="blue")
plt.axhline(y=40, color ="red", linestyle = "-", label = "Norgespris")
plt.xticks(rotation=90)
plt.title("Forbruk time for time", fontsize=14)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Kostnad i Øre (NOK)", fontsize=12)
plt.show()





#def betale_kr(spotpris,forbruk):
    #print("forbruk")



#def strømstøtte_kr(spotpris,forbruk, ordning):
    #print(ordning)


#teste meg frem
#strømstøtte_kr(1,2,3)
#betale_kr(5,6)


#uki pseudokode

#ta inn spotpris og forbruk for hver time og gange sammen
#printe original
#legge inn en if løkke;
#hvis spotpris e over 75(?) øre, så forsvinner 90% av prisen videre
#printe dette fint og
#og sammenligne med printe original, får derfra ka strømstøtten blir

#gjøre det samme med 40øre og forbruk og se ka forsjellen blir


#ELLER bare ta for løkke og lagre alle spotpriser som går over grensen,
# og regne på disse og skrive ut, og derfra gange med forbruk? for å så se kor mye de får i støtte




