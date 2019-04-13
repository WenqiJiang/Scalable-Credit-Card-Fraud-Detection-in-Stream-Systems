import numpy as np
from sklearn.linear_model import LogisticRegression

X_train = np.load("./data/subsamp_data/processed_X_train.npy")
X_test = np.load("./data/subsamp_data/processed_X_test.npy")
X_val = np.load("./data/subsamp_data/processed_X_val.npy")
y_train = np.load("./data/subsamp_data/processed_y_train.npy")
y_test = np.load("./data/subsamp_data/processed_y_test.npy")
y_val = np.load("./data/subsamp_data/processed_y_val.npy")

clf = LogisticRegression(C=100).fit(X_train, y_train)

print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on validation set: {:.2f}'
     .format(clf.score(X_val, y_val)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))