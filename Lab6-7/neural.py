import random

import numpy as np

wz_in = []
wz_out = []
Nwz = 0

file = open("./dane", "r")
n_in = int(file.readline().rstrip('\n'))
n_out = int(file.readline().rstrip('\n'))
n_hid = int(file.readline().rstrip('\n'))
eta = float(file.readline().rstrip('\n'))
alfa = float(file.readline().rstrip('\n'))
bias = float(file.readline().rstrip('\n'))
zm_bias = file.readline().rstrip('\n')
n_epoch = file.readline().rstrip('\n')
funkcja = file.readline().rstrip('\n')
beta = float(file.readline().rstrip('\n'))
losowo = file.readline().rstrip('\n')
file.close()

weight1 = [[0 for x in range(10)] for y in range(10)]
old_weight_change1 = [[0 for x in range(100)] for y in range(100)]
weight_change1 = [[0 for x in range(100)] for y in range(100)]
weight2 = [[0 for x in range(10)] for y in range(10)]
weight_change2 = [[0 for x in range(100)] for y in range(100)]
old_weight_change2 = [[0 for x in range(100)] for y in range(100)]

sum1 = 0.0
sum2 = 0.0
sum3 = 0.0

scale = 1.0

RMS = 0.0


def g(e):
    global weight1, weight2, wz_in, wz_out, Nwz, sum1, sum2, sum3
    if funkcja == 'tan':
        return np.tanh(beta * e)
    if funkcja == 'sig':
        return 1.0 / (1.0 + np.exp(-beta * e))
    if funkcja == 'lin':
        return beta * e
    if funkcja == 'gau':
        return np.exp(-e * e / 2.0)
    else:
        print("error")
        return 0


def g1(e):
    global weight1, weight2, wz_in, wz_out, Nwz, sum1, sum2, sum3
    if funkcja == "tan":
        return float(beta) * (1.0 - g(e) * g(e))
    if funkcja == 'sig':
        return 2.0 * beta * g(e) * (1.0 - g(e))
    if funkcja == 'lin':
        return beta
    if funkcja == 'gau':
        return -e * np.exp(-e * e / 2.0)

    else:
        print("error")
        return 0


def print_wyn(epoch, RMS):
    global weight1, weight2, wz_in, sum1, sum2, sum3, scale
    print("epoch: {}, RMS: {:e}".format(epoch, RMS))


def test():
    global weight1, weight2, wz_in, wz_out, Nwz, sum1, sum2, sum3, scale, RMS
    V0 = np.tile(0.0, 100)
    V1 = np.tile(0.0, 100)
    V2 = np.tile(0.0, 100)
    print("Testing Time")
    V0[0] = 1.0
    V1[0] = 1.0
    Tin = []
    Tout = []

    tesf = open("./test")

    for count, line in enumerate(tesf, start=0):
        if count % 2 == 0:
            Tin.append(line.rstrip('\n').split(' '))
            # Nwz += 1
        if count % 2 == 1:
            Tout.append(line.rstrip('\n'))

    tesf.close()

    for number in range(0, len(Tin)):
        # Przepisanie danej testowej na wejscie sieci
        for i in range(0, n_in):
            V0[i] = float(Tin[number][i]) / scale
        #   Warstwa ukryta
        for i in range(0, n_hid):
            sum1 = 0.0
            for j in range(0, n_in):
                sum1 += float(weight1[i][j]) * V0[j]
            V1[i] = g(sum1)
        #   Warstwa wyjsciowa
        for i in range(0, n_out):
            sum2 = 0.0
            for j in range(n_hid):
                sum2 += float(weight2[i][j]) * V1[j]
            V2[number] = g(sum2)
        for j in range(0, n_out):
            print("Real: {}, Predicted: {:2f}".format(Tout[number],
                                                      V2[number] * scale))


def learning():
    global weight1, weight2, wz_in, wz_out, Nwz, sum1, sum2, sum3, scale, RMS
    V0 = np.tile(0.0, 100)
    V1 = np.tile(0.0, 100)
    V2 = np.tile(0.0, 100)
    delta1 = np.tile(0.0, 4)
    delta2 = np.tile(0.0, 4)
    pattern_tab = np.tile(0.0, 100)

    # Jezeli przesuniecie (bias) ma byc ustalone, to petle obliczajace
    # blad i korygujace wagi wykonujemy od 1, jezeli dopuszczalne sa
    # zmiany tych wartosci, petle wykonujemy od 0.
    if zm_bias:
        prog = 0
    else:
        prog = 1

    # Wczytanie wzorcow
    f = open('./learning_data', "r")
    for count, line in enumerate(f, start=0):
        if count % 2 == 0:
            wz_in.append(line.rstrip('\n').split(' '))
            Nwz += 1
        if count % 2 == 1:
            wz_out.append(float(line.rstrip('\n')))
    f.close()
    Nwz -= 1

    # Szukanie najwiekszej liczby wsrod wzorcow (co do modulu)
    scale = 1.0
    for i in range(0, Nwz):
        for j in range(0, n_in):
            if abs(float(wz_in[i][j])) > scale:
                scale = abs(float(wz_in[i][j]))
        for j in range(0, n_out):
            if abs(float(wz_out[i])) > scale:
                scale = abs(float(wz_out[i][j]))

    # Szukanie najwiekszej liczby wsrod danych testowych
    if (funkcja != "liniowa"):
        tfile = open("./test")
        for value in tfile:
            for val in value.rstrip('\n').split(' '):
                if abs(float(val)) > scale:
                    scale = abs(float(val))
        tfile.close()

    # Transformacja do przedzialu (-1,1)
    for i in range(0, Nwz):
        for j in range(0, n_in):
            wz_in[i][j] = float(wz_in[i][j]) / scale
        for j in range(0, n_out):
            wz_out[j] = float(wz_out[j]) / scale

    print("Learning Time")

    random.seed(1231)
    for i in range(0, n_hid):
        for j in range(0, n_in):
            los = random.uniform(0, 1)
            weight1[i][j] = los / 10.0 - 0.05
    for i in range(0, n_out):
        for j in range(0, n_hid):
            los = random.uniform(0, 1)
            weight2[i][j] = los / 10.0 - 0.05

    # Wyzerowanie tablicy "poprzednich" zmian wag (momentum)
    # Warstwa ukryta
    for i in range(0, n_hid):
        for j in range(0, n_in):
            old_weight_change1[i][j] = 0.0
    # Warstwa wyjsciowa
    for i in range(0, n_out):
        for j in range(0, n_hid):
            old_weight_change2[i][j] = 0.0

    # Uwzglednienie przesuniecia (bias) - dodatkowe wejscie o numerze 0
    # i nastepujacych wartosciach waga(i,0)=bias; V(0)=1.0
    # Warstwa wejsciowa
    for i in range(0, n_in):
        weight1[i][0] = bias

    # Warstwa ukryta
    for i in range(0, n_hid):
        weight2[i][0] = bias
    V0[0] = 1.0
    V1[0] = 1.0

    # Glowna petla procedury
    for epoch in range(0, int(n_epoch)):
        RMS = 0.0

        # Petla po wszystkich wzorcach
        for pattern in range(0, Nwz):

            # Jezeli losowo=.TRUE. to wzorce podajemy na wejscie w losowej kolejnosci
            # Jezeli losowo=.FALSE. to wzorce podajemy sekwencyjnie na wejscie
            if losowo != 0:
                los = random.uniform(0, 1)
                pattern_number = int(los * Nwz) + 1
                ifWas = False
                for j in range(0, pattern - 1):
                    if pattern_tab[j] == pattern_number:
                        ifWas = True

                if ifWas:
                    break

                pattern_tab[pattern] = pattern_number
            else:
                pattern_number = pattern

            # Petla po neuronach warstwy wejsciowej (Nin neuronow X 1 Wejscie)
            for i in range(0, n_in):
                # przepisanie wzorcow do warstwy wejsciowej
                V0[i] = wz_in[pattern_number][i]

            #  Petla po neuronach warstwy ukrytej (Nhid neuronow X Nin+1 wejsc)
            for i in range(0, n_hid):
                sum1 = 0.0
                # Petla po wejsciach neuronow warstwy ukrytej
                for j in range(0, n_in):
                    sum1 = sum1 + float(weight1[i][j]) * V0[j]
                V1[i] = g(sum1)

            # Petla po neuronach warstwy wyjsciowej (Nout neuronow X Nhid+1 wejsc)
            for i in range(0, n_out + 1):
                sum2 = 0.0
                # Petla po wejsciach neuronow warstwy wyjsciowej
                for j in range(0, n_hid):
                    sum2 += float(weight2[i][j]) * V1[j]
                V2[i] = g(sum2)

            # print(V2)
            # if epoch % 10000 == 0.0:
            #    print_wyn(epoch, V2, RMS, pattern, Nwz, n_out, wz_out,
            #              pattern_number)

            if epoch == n_epoch and pattern == Nwz:
                break

            #   Obliczenie bledow i korekcja wag

            #   Blad na wyjsciu warstwy wyjsciowej
            for i in range(0, n_out):
                delta2[i] = g1(sum2) * (float(wz_out[pattern_number]) - V2[i])

            #   Blad na wyjsciu warstwy ukrytej
            for i in range(0, n_hid):
                sum3 = 0.0
                for j in range(prog, n_out + 1):
                    sum3 += float(weight2[j][i]) * float(delta2[j])
                delta1[i] = g1(sum1) * sum3

            #   Korekcja wag
            #   Warstwa wyjsciowa
            for i in range(0, n_out):
                for j in range(prog, n_hid):
                    weight_change2[i][j] = eta * delta2[i] * V1[j]
                    weight2[i][j] = weight2[i][j] + weight_change2[i][
                        j] + alfa * old_weight_change2[i][j]

        # Warstwa ukryta
        for i in range(0, n_hid):
            for j in range(prog, n_in):
                weight_change1[i][j] = eta * delta1[i] * V0[j]
                weight1[i][j] = weight1[i][j] + weight_change1[i][j] + alfa * \
                                old_weight_change1[i][j]

        #   Blad sredni kwadratowy
        for i in range(0, n_out):
            RMS = RMS + pow(
                wz_out[pattern_number] * scale - V2[i] * scale,
                2)
        RMS = RMS / 2.0

        #   Uaktualnienie tablicy "poprzednich" zmian wag (momentum)
        #   Warstwa ukryta
        for i in range(0, n_hid):
            for j in range(prog, n_in):
                old_weight_change1[i][j] = weight_change1[i][j]
        #   Warstwa wyjsciowa
        for i in range(0, n_out):
            for j in range(prog, n_hid):
                old_weight_change2[i][j] = weight_change2[i][j]

        if (epoch % 10000) == 0.0:
            print_wyn(epoch, RMS)


learning()
test()
