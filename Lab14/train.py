from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

digits = load_digits()

x = digits.data
y = digits.target

x_train, x_test, y_train, y_test = train_test_split(x, y)

# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
clf = RandomForestClassifier(n_estimators=10, random_state=20701)
clf.fit(x_train, y_train)

pred = clf.predict(x_test)

hit = 0
miss = 0
for i in range(0, len(y_test)):
    actual_class = y_test[i]
    pred_class = pred[i]
    if actual_class != pred_class:
        miss += 1
        print("Actual: {}".format(actual_class), end=" - ")
        print("Predicted: {}".format(pred_class))
    else:
        hit += 1

print("\nGood predictions : {}".format(hit))
print("Bad predictions : {}".format(miss))
print("Total accuracy: \t", clf.score(x_test, y_test))
