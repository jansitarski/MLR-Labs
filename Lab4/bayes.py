'''The naive Bayes method. See attached .yaml files to see how to provide input data'''

import yaml
from select_file import select_file

filename = select_file("yaml")

file = open(filename, "r", encoding='utf8')
data = yaml.safe_load(file)

print("\nHipoteza, prawdopodobieństwo a priori:")
for h in data["Hypotheses"]:
    print("{}, {}%".format(h["name"], h["prob"] * 100))

print("\nPOJEDYNCZE FAKTY")

# Calculate probability of facts
Pr_f = []
for fact in data["Facts"]:
    sum = 0
    for index, h in enumerate(data["Hypotheses"]):
        sum = sum + h["prob"] * fact["prob"][index]
    Pr_f.append([fact["name"], sum])

print("Fakt, prawdopodobieństwo:")
for x in Pr_f:
    print("{}, {:.2f}%".format(x[0], x[1] * 100))

# Calculate probability of hypothesis under a single fact
Pr_h_f = []
for indexh, h in enumerate(data["Hypotheses"]):
    for indexf, fact in enumerate(data["Facts"]):
        pr = h["prob"] * fact["prob"][indexh] / Pr_f[indexf][1]
        Pr_h_f.append([h["name"], fact["name"], pr])

print("\nHipoteza, fakt, prawdopodobieństwo:")
for x in Pr_h_f:
    print("{}, {}, {:.2f}%".format(x[0], x[1], x[2] * 100))

# Calculate probability of hypothesis under a list of facts
print("\nKILKA FAKTÓW JEDNOCZEŚNIE")
print("Fakty:")
for index, fact in enumerate(data["Facts"]):
    print(index, fact["name"])

# Read list of facts to be included into the calculations
ok = False
while not ok:
    selected_facts_indexes = sorted(input("Podaj numery faktów, oddzielając je spacjami: ").split())
    # Convert to int
    selected_facts_indexes = [int(i) for i in selected_facts_indexes]
    # Remove duplicates
    selected_facts_indexes = list(dict.fromkeys(selected_facts_indexes))
    # Check if provided numbers are allowed
    ok = set(selected_facts_indexes).issubset(range(len(data["Facts"])))

selected_facts = [data["Facts"][int(i)]["name"] for i in selected_facts_indexes]

'''
Zadanie:
a) uzupełnić program tak, aby uwzględniał dowolne zestawy faktów
   wg. wzoru w żółtej obwódce w pliku PDF,
b) utworzyć nowy plik z definicjami faktów, który mógłby posłużyć do wnioskowania o chorobach: 
   COVID 19 / grypa / przeziębienie / inna choroba.
'''

# Denominator
summ = 0
denominator = 1
for ih, h in enumerate(data["Hypotheses"]):
    temp = 1
    temp *= h["prob"]
    for f in data["Facts"]:
        temp *= f["prob"][ih]
    summ += temp
denominator = summ

# Numerator
for ih, h in enumerate(data["Hypotheses"]):
    numerator = 1
    numerator *= h["prob"]

    for f in data["Facts"]:
        numerator *= f["prob"][ih]

    # Calculate and print the final result
    pr = numerator / denominator
    print("Prawdopodobieństwo hipotezy {} przy uwzględnieniu faktów ({}) wynosi: {:.2f}%." \
          .format(h["name"], ', '.join(selected_facts), pr * 100))
