###############################################################################
#
# Python 3
#
# Wcięcia realizowane są czterema spacjami.
#
# Doczytanie bibliotek numpy i matplotlib:
# pip install numpy
# pip install matplotlib
#
# Uruchamianie skryptu:
# python zadanie.py
# albo wymuszając Pythona 3 gdy nie jest on domyślny:
# py -3 zadanie.py
#
###############################################################################
#
# Plik dane.csv zawiera dane zbierane na węźle ciepłowniczym przez
# przedsiębiorstwo dostarczające ciepło do budynku (patrz opisy kolumn w pliku).
# Niniejszy skrypt dokonuje podstawowej analizy tych danych.
#
# A.
# Wczytanie obserwacji dla wybranych zmiennych.
#
# B.
# Sprawdzenie podstawowych statystyk dla poszczególnych zmiennych.
# Wykreślenie histogramów.
#
# C.
# Identyfikacja zmiennych, w których występują potencjalnie błędne dane (obserwacje)
# lub braki danych. Naprawa danych.
#
# D.
# Obliczenie unormowanych korelacji pomiędzy poszczególnymi zmiennymi.
#
# E.
# Przeprowadzenie regresji liniowej dla wybranych zmiennych, wraz z wykresami.
#
###############################################################################
#
# Należy wykonać zadania na sumarycznie co najmniej 4 punkty
#
###############################################################################


import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#######################
# A. Wczytanie danych #
#######################

przeplyw = []  # Przepływ wody przez węzeł
temp_zasilania = []  # Temperatura wody na wejściu do węzła
temp_powrotu = []  # Temperatura wody na wyjściu z węzła
roznica_temp = []  # Różnica temperatur, wynikająca z oddanej energii w węźle
moc = []  # Moc oddana w węźle

plik = open('dane.csv', 'rt')
dane = csv.reader(plik, delimiter=',')
next(dane)  # Opuszczamy pierwszy wiersz (nagłówek)
for obserwacja in dane:  # Iterujemy po poszczególnych obserwacjach
    przeplyw.append(float(obserwacja[6]))
    temp_zasilania.append(float(obserwacja[7]))
    temp_powrotu.append(float(obserwacja[8]))
    roznica_temp.append(float(obserwacja[9]))
    moc.append(float(obserwacja[12]))
plik.close()

### ZADANIE (0.5p.) ###
# Dane w listach są ułożone od najnowszych do najstarszych.
# Odwrócić dane na listach tak, żeby były ułożone chronologicznie.
### KONIEC ###

# Tworzymy słownik: kluczem jest nazwa zmiennej a wartością - zmienna
przeplyw.reverse()
temp_zasilania.reverse()
temp_powrotu.reverse()
roznica_temp.reverse()
moc.reverse()

zmienne = {"Przepływ": przeplyw,
           "Temperatura zasilania": temp_zasilania,
           "Temperatura powrotu": temp_powrotu,
           "Różnica temperatur": roznica_temp,
           "Moc": moc
           }

######################################
# B. Podstawowe statystyki i wykresy #
######################################

# Iterujemy po słowniku, wyświetlając statystyki dla poszczególnych zmiennych
for nazwa, zmienna in zmienne.items():
    print("\nZmienna:", nazwa)
    print("MIN:", min(zmienna))
    print("MAX:", max(zmienna))
    print("ŚREDNIA:", np.mean(zmienna))
    print("MEDIANA:", np.median(zmienna))
    print("ZAKRES:", np.ptp(zmienna))
    print("ODCHYLENIE STANDARDOWE:", np.std(zmienna))
    print("WARIANCJA:", np.var(zmienna))
    print("PERCENTYL 90%:", np.percentile(zmienna, 90))
    print("HISTOGRAM:", np.histogram(zmienna))

    # Wykres - histogram
    plt.hist(zmienna, 100)
    plt.title('Histogram dla: ' + nazwa)
    plt.xlabel('Przedział')
    plt.ylabel('Liczba obserwacji')
    plt.show()

############################################
# C. Analiza anomalii i czyszczenie danych #
############################################

# Zidentyfikowaliśmy problem - "dziwne", znacząco za duże niektóre wartości dla zmiennych:


### ZADANIE (1p.) ###
# Zrealizować automatyczne dodawanie "podejrzanych" zmiennych do słownika "zmienne_do_naprawienia",
# na podstawie analizy statystyk danej zmiennej.
### KONIEC ###

zmienne_do_naprawienia = {}
for nazwa, zmienna in zmienne.items():
    if (max(zmienna) > 10000):
        zmienne_do_naprawienia[nazwa] = zmienna

print("\nCZYSZCZENIE DANYCH")

for nazwa, zmienna in zmienne_do_naprawienia.items():

    Q1 = np.percentile(zmienna, 25)
    Q3 = np.percentile(zmienna, 75)
    IQR = Q3 - Q1
    rang = 1.5 * IQR
    lowerFence = Q1 - rang
    upperFence = Q3 + rang

    for index, wartosc in enumerate(zmienna):  # Iterujemy po wszystkich obserwacjach
        if wartosc > upperFence or wartosc < lowerFence:
            print("Dla zmiennej {} pod indeksem {} znaleziono anomalię o wartości {}".format(nazwa, index, wartosc))
            # Wstawiamy medianę:
            mediana = np.median(zmienna)
            print("Naprawiam. Stara wartość: {}, nowa wartość: {}".format(zmienna[index], mediana))
            zmienna[index] = mediana

### ZADANIE (1p.) ###
# Znaleźć inną metodę wyznaczania progu anomalii w powyższej pętli tak, aby nie była to
# "hardkodowana" wartość 10000, ale liczba wyznaczana indywidualnie dla każdej zmiennej.
# Jedna z metod - metoda IQR: https://online.stat.psu.edu/stat200/lesson/3/3.2
# Inna podpowiedź: https://mateuszgrzyb.pl/3-metody-wykrywania-obserwacji-odstajacych-w-python/ 
### KONIEC ###

# Statystyki dla naprawionych zmiennych
for nazwa, zmienna in zmienne.items():
    print("\nZmienna (naprawiona):", nazwa)
    print("MIN:", min(zmienna))
    print("MAX:", max(zmienna))
    print("ŚREDNIA:", np.mean(zmienna))
    print("MEDIANA:", np.median(zmienna))
    print("ZAKRES:", np.ptp(zmienna))
    print("ODCHYLENIE STANDARDOWE:", np.std(zmienna))
    print("WARIANCJA:", np.var(zmienna))
    print("PERCENTYL 90%:", np.percentile(zmienna, 90))
    print("HISTOGRAM:", np.histogram(zmienna))

    plt.hist(zmienna, 100)
    plt.title('Histogram dla: ' + nazwa)
    plt.xlabel('Przedział')
    plt.ylabel('Liczba obserwacji')

    #   for nazwa, zmienna in zmienne.items():
    #       plt.text(3, 35, "MIN: "+ str(min(zmienna)))
    #       plt.text(3, 35, "MAX: "+ str(max(zmienna)))
    #       plt.text(3, 35, "ŚREDNIA: "+ str(np.mean(zmienna)))
    #       plt.text(3, 35, "MEDIANA: "+ str(np.median(zmienna)))
    #       plt.text(3, 35, "ZAKRES: "+ str(np.ptp(zmienna)))
    #       plt.text(3, 35, "ODCHYLENIE STANDARDOWE: "+ str(np.std(zmienna)))
    #       plt.text(3, 35, "WARIANCJA: "+ str(np.var(zmienna)))
    #       plt.text(3, 35, "PERCENTYL 90%: "+ str(np.percentile(zmienna, 90)))
    #       plt.text(3, 35, "HISTOGRAM: "+ str(np.histogram(zmienna)))

    plt.savefig('Histogram ' + nazwa + ".pdf")
    plt.show()

### ZADANIE (1p.) ###
# Zapisać powyższe statystyki i wykresy do plików PDF, osobnych dla poszczególnych zmiennych
# (można wykorzystać dowolny moduł/bibliotekę).
### KONIEC ###


#########################################
# D. Badanie korelacji między zmiennymi #
#########################################

print()
print("KORELACJE")


# Piszemy funkcję, która zwróci korelację unormowaną między zestawami danych
def ncorrelate(a, b):
    '''Funkcja zwraca unormowaną wartość korelacji'''
    a = (a - np.mean(a)) / (np.std(a) * len(a))
    b = (b - np.mean(b)) / np.std(b)
    return np.correlate(a, b)[0]


weak_corr = {}
strong_corr = {}

for nazwa, zmienna in zmienne.items():
    for nazwa2, zmienna2 in zmienne.items():
        if ncorrelate(zmienna, zmienna2) > 2:
            strong_corr[nazwa] = nazwa2
        else:
            weak_corr[nazwa] = nazwa2

### ZADANIE (0.5p.) ###
# Zademonstrować działanie funkcji ncorrelate() na przykładach:
# a. dwóch list zawierających dane silnie skorelowane
# b. dwóch list zawierające dane słabo skorelowane
# Listy należy generować automatycznie
### KONIEC ###

### ZADANIE (0.5p.) ###
# Poszukać funkcji z pakietu numpy, która wykonuje identyczne zadanie jak
# funkcja ncorrelate() i ją wykorzystać.
### KONIEC ###

# Badamy korelacje między wszystkimi (różnymi od siebie) zmiennymi
correl = {}
corrList =[]
for nazwa1, zmienna1 in zmienne.items():
    for nazwa2, zmienna2 in zmienne.items():
        if nazwa1 != nazwa2:
            print("Korelacja między", nazwa1, "a", nazwa2, "wynosi:", end=" ")
            print(ncorrelate(zmienna1, zmienna2))

### ZADANIE (1p.) ###
# Zebrać powyższe wyniki korelacji w dwuwymiarowej liście postaci:
# [[zmienna1, zmienna2, korelacja], [..., ..., ...], ... ] tak, aby elementy tej listy
# były posortowane malejąco wg. wartości korelacji.
### KONIEC ###

# Przykładowe wykresy

# 1. Zmienne z dużą korelacją dodatnią: moc, przeplyw

# Wykres liniowy
plt.plot(range(len(moc)), moc, "x", label="Moc")
plt.plot(range(len(przeplyw)), przeplyw, "+", label="Przepływ")
plt.title("Duża korelacja dodatnia")
plt.xlabel('Numer obserwacji')
plt.legend()
plt.show()

# Dla lepszej ilustracji: wycinek danych.
# Zmienna moc przemnożnona przez 10, aby lepiej było widać korelację.
plt.plot(range(len(moc[1000:1100])), [i * 10 for i in moc[1000:1100]], label="Moc")
plt.plot(range(len(przeplyw[1000:1100])), przeplyw[1000:1100], label="Przepływ")
plt.title("Duża korelacja dodatnia; Moc przemnożona przez 10")
plt.xlabel('Numer obserwacji')
plt.legend()
plt.show()

# Wykres zależności przeplyw od moc
plt.plot(moc, przeplyw, '.')
plt.title("Duża korelacja dodatnia")
plt.xlabel('Moc')
plt.ylabel('Przeplyw')
plt.show()

# 2. Zmienne skorelowane ujemnie: roznica_temp, temp_powrotu

# Wykres liniowy
plt.plot(range(len(roznica_temp)), roznica_temp, "x", label="Różnica temperatur")
plt.plot(range(len(temp_powrotu)), temp_powrotu, "+", label="Temperatura powrotu")
plt.title("Średnia korelacja ujemna")
plt.xlabel('Numer obserwacji')
plt.legend()
plt.show()

# Dla lepszej ilustracji: wycinek danych
plt.plot(range(len(roznica_temp[1000:1100])), roznica_temp[1000:1100], label="Różnica temperatur")
plt.plot(range(len(temp_powrotu[1000:1100])), temp_powrotu[1000:1100], label="Temperatura powrotu")
plt.title("Średnia korelacja ujemna")
plt.xlabel('Numer obserwacji')
plt.legend()
plt.show()

# Wykres zależności temp_powrotu od roznica_temp
plt.plot(roznica_temp, temp_powrotu, '.')
plt.title("Średnia korelacja ujemna")
plt.xlabel('Różnica temperatur')
plt.ylabel('Temperatura powrotu')
plt.show()

#######################
# E. Regresja liniowa #
#######################

# Analiza przeprowadzona tylko dla jednej zmiennej, temp_zasilania

print()
print("REGRESJA LINIOWA")
# Wybieramy zmienną temp_zasilania w funkcji numeru obserwacji
x = range(len(temp_zasilania))
y = temp_zasilania
# Liczymy współczynniki regresji - prostej
a, b = np.polyfit(x, y, 1)  # Wielomian 1 rzędu - prosta
print("Wzór prostej: y(x) =", a, "* x +", b)
# Wyliczamy punkty prostej otrzymanej w wyniku regresji
yreg = [a * i + b for i in x]
# Wykresy
plt.plot(x, y, label="Temperatura zasilania")
plt.plot(x, yreg, label="Wynik regresji")
plt.title("Regresja liniowa dla całosci danych zmiennej temp_zasilania")
plt.xlabel('Numer obserwacji')
plt.legend()
plt.show()

print()
print("REGRESJA WIELOMIANOWA")
# Wybieramy zmienną temp_zasilania w funkcji numeru obserwacji
x = range(len(temp_zasilania))
y = temp_zasilania
# Liczymy współczynniki regresji - prostej
a, b, c = np.polyfit(x, y, 2)  # Wielomian 1 rzędu - prosta
print("Wzór krzywej: y(x) =", a, "* x *x  +", b, "* x +", c)
# Wyliczamy punkty prostej otrzymanej w wyniku regresji
yreg = [a * i * i + b * i + c for i in x]
# Wykresy
plt.plot(x, y, label="Temperatura zasilania")
plt.plot(x, yreg, label="Wynik regresji")
plt.title("Regresja liniowa dla całosci danych zmiennej temp_zasilania")
plt.xlabel('Numer obserwacji')
plt.legend()
plt.show()

### ZADANIE (1.5p.) ###
# Z wykresu widać, że regresja liniowa dla całości zmiennej temp_zasilania słabo się sprawdza.
# Wynika to z tego, że inaczej dane rozkładają się w róznych porach roku.
# Należy więc podzielić dane na kilka podzakresów i regresję wykonać osobno
# dla każdego z podzakresu. Narysować odpowiedni wykres.
### KONIEC ###

# Regresja liniowa dla zmiennych z dużą korelacją dodatnią: moc, przeplyw
a, b = np.polyfit(moc, przeplyw, 1)  # Wielomian 1 rzędu - prosta
yreg = [a * i + b for i in moc]
# Wykresy
plt.plot(moc, przeplyw, ".")
plt.plot(moc, yreg, label="Wynik regresji")
plt.title("Regresja liniowa")
plt.xlabel('Moc')
plt.ylabel('Przepływ')
plt.legend()
plt.show()

# Regresja liniowa dla zmiennych ze słabą korelacją ujemną: roznica_temp, temp_powrotu
a, b = np.polyfit(roznica_temp, temp_powrotu, 1)  # Wielomian 1 rzędu - prosta
yreg = [a * i + b for i in roznica_temp]
# Wykresy
plt.plot(roznica_temp, temp_powrotu, ".")
plt.plot(roznica_temp, yreg, label="Wynik regresji")
plt.title("Regresja liniowa")
plt.xlabel('Różnica temperatur')
plt.ylabel('Temperatura powrotu')
plt.legend()
plt.show()

# Predykcja danych z losowej listy
roznica_temp = []
import random

for i in range(20):
    roznica_temp.append(random.randint(0, 100))

# Wyliczenie wyników na podstawie regresji i zapis do listy
temp_powrotu = [[i, a * i + b] for i in roznica_temp]
print("Wyniki predykcji temp_powrotu(roznica_temp):", temp_powrotu)

### ZADANIE (0.5p.) ###
# Zapisać wyniki powyższej predykcji do pliku JSON o nazwie predykcja.json
### KONIEC ###
