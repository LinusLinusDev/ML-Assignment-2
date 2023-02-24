# -------------------------------------------------------------------------
# AUTHOR: Linus Palm
# FILENAME: knn.py
# SPECIFICATION: This program determines the leave-one-out cross-validation
# error rate for 1NN to the points from binary_points.csv.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 6 hours
# -----------------------------------------------------------

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to
# work here only with standard vectors and arrays

# importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

# reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            db.append(row)

errors = 0

# loop your data to allow each instance to be your test set
for test in db:
    # add the training features to the 2D array X removing the instance that will be used for testing in this
    # iteration. For instance, X = [[1, 3], [2, 1,], ...]]. Convert each feature value to float to avoid warning
    # messages

    # transform the original training classes to numbers and add to the vector Y removing the instance that will be
    # used for testing in this iteration. For instance, Y = [1, 2, ,...]. Convert each feature value to float to
    # avoid warning messages

    X = []
    Y = []
    for training in db:
        if training == test:
            continue
        Y.append(1.0 if training[2] == '-' else 2.0)
        X.append(list(map(lambda x: float(x), training[0:2])))

    # store the test sample of this iteration in the vector testSample
    Y_test = 1.0 if test[2] == '-' else 2.0
    X_test = list(map(lambda x: float(x), test[0:2]))

    # fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    # use your test sample in this iteration to make the class prediction. For instance:
    class_predicted = clf.predict([X_test])[0]

    # compare the prediction with the true label of the test instance to start calculating the error rate.
    if class_predicted != Y_test:
        errors += 1

# print the error rate
print("error rate:", errors / len(db))
