# -------------------------------------------------------------------------
# AUTHOR: Linus Palm
# FILENAME: decision_tree_2.py
# SPECIFICATION: This program generates and tests three different decision trees using three
# different training sets and one test set. At the end, the average accuracy of each training set is output.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 6 hours
# -----------------------------------------------------------

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to
# work here only with standard dictionaries, lists, and arrays

# importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    # reading the training data in a csv file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0:  # skipping the header
                dbTraining.append(row)

    # transform the original categorical training features to numbers and add to the 4D array X. For instance Young =
    # 1, Prepresbyopic = 2, Presbyopic = 3 so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]

    # transform the original categorical training classes to numbers and add to the vector Y. For instance Yes = 1,
    # No = 2, so Y = [1, 1, 2, 2, ...]

    valuetonumber = {
        'Young': 1,
        'Prepresbyopic': 2,
        'Presbyopic': 3,
        'Myope': 1,
        'Hypermetrope': 2,
        'Yes': 1,
        'No': 2,
        'Reduced': 1,
        'Normal': 2
    }

    for sample in dbTraining:
        numbersample = []
        for value in sample:
            numbersample.append(valuetonumber[value])
        Y.append(numbersample.pop())
        X.append(numbersample)

    predictedCorrect = 0

    # loop your training and test tasks 10 times here
    for i in range(10):
        # fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
        clf = clf.fit(X, Y)

        # read the test data and add this data to dbTest
        dbTest = []
        with open('contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for j, row in enumerate(reader):
                if j > 0:  # skipping the header
                    dbTest.append(row)

        for data in dbTest:
            # transform the features of the test instances to numbers following the same strategy done during
            # training, and then use the decision tree to make the class prediction. For instance: class_predicted =
            # clf.predict([[3, 1, 2, 1]])[0] where [0] is used to get an integer as the predicted class label so that
            # you can compare it with the true label
            X_test = []
            Y_test = None
            for value in data:
                X_test.append(valuetonumber[value])
            Y_test = X_test.pop()
            # compare the prediction with the true label (located at data[4]) of the test instance to start
            # calculating the accuracy.
            if Y_test == clf.predict([X_test])[0]:
                predictedCorrect += 1

    # find the average of this model during the 10 runs (training and test set)
    average = predictedCorrect / 10

    # print the average accuracy of this model during the 10 runs (training and test set).
    # your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    print("final accuracy when training on", ds, ":", average / 8)
