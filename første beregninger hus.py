2
#her skal vi prøve å beregne et hus
#vi velger oss et random hus som testobjekt


def betale_kr(spotpris,forbruk):
    print("forbruk")



def strømstøtte_kr(spotpris,forbruk, ordning):
    print(ordning)


#teste meg frem
strømstøtte_kr(1,2,3)
betale_kr(5,6)


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



#importere data
import pandas as pd

data = pd.read_excel(r'C:\Users\kristinemoen\Skrivebord\Exp_56.xlsx') #fikse opp i denne Kristine
print(data)
