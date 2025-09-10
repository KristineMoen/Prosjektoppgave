#importere data
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel(r'/Users/kristinemoen/Desktop/Exp_56.xlsx') #Denne filen er Exp_56, et rekkehus i Oslo i Phase 1
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
print(data)

#her skal vi prøve å beregne et hus (rekkehus i første omgang, i Oslo)
#vi velger oss et random hus som testobjekt

data["Cost per Hour"] = data["Demand_kWh"] * data["Price/kWh"]
print(data)

#Stoltediagram

plt.figure(figsize=(12, 6))
plt.bar(data.index + 1, data["Cost per Hour"], color ="blue")
plt.title("Forbruk time for time", fontsize=14)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Kostnad (NOK)", fontsize=12)
plt.show()

#annen måte å lage stolpediagrammet på
#dager = len(data)//24
#fig, axes = plt.subplots(dager, 1, figsize =(14, 4*dager), sharex = True)
#for i in range(dager):
    #plt.figure()
    #dag_data = data.iloc[i*24:(i+1)*24]
    #axes[i].bar(range(1,25), dag_data["Cost per Hour"], color = "blue")
    #axes[i].set_title(f"Dag {i+1}")
    #axes[i].set_xlabel("Timer")
    #axes[i].set_ylabel("Kostander i NOK")


#plt. show()



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




