# -------------------------------------------------------------------------
# AUTHOR: Linus Palm
# FILENAME: naive_bayes.py
# SPECIFICATION: This program reads the file weather_training.csv and output the classification of each
# test instance from the file weather_test.csv if the classification confidence is >= 0.75.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 6 hours
# -----------------------------------------------------------

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to
# work here only with standard dictionaries, lists, and arrays

# importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

# reading the training data in a csv file
Header = ""
dbTraining = []
with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            dbTraining.append(row)
        else:
            Header = row
            Header.append('Confidence')

# transform the original training features to numbers and add them to the 4D array X.
# For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]

# transform the original training classes to numbers and add them to the vector Y.
# For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
X = []
Y = []

valuetonumber = {
    'Sunny': 1,
    'Overcast': 2,
    'Rain': 3,
    'Hot': 1,
    'Cool': 2,
    'Mild': 3,
    'Normal': 1,
    'High': 2,
    'Weak': 1,
    'Strong': 2,
    'No': 1,
    'Yes': 2
}

for sample in dbTraining:
    numbersample = []
    for idx, value in enumerate(sample):
        if idx == 0:
            continue
        numbersample.append(valuetonumber[value])
    Y.append(numbersample.pop())
    X.append(numbersample)

# fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

# reading the test data in a csv file
dbTest = []
with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            dbTest.append(row)

X_test = []

for sample in dbTest:
    numbersample = []
    for idx, value in enumerate(sample):
        if idx == 0 or value == '?':
            continue
        numbersample.append(valuetonumber[value])
    X_test.append(numbersample)

# printing the header of the solution
layout = "{:<5}{:<10}{:<13}{:<10}{:<8}{:<12}{:<12}"
print(layout.format(*Header))

# use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
for idx, sample in enumerate(X_test):
    predictedClass = clf.predict([sample])[0]
    confidence = clf.predict_proba([sample])[0][predictedClass - 1]
    if confidence >= 0.75:
        print(layout.format(*dbTest[idx][0:5], 'No' if predictedClass == 1 else 'Yes', round(confidence, 2)))
