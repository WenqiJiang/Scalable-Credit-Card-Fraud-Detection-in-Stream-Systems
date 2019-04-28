import numpy as np
from sklearn.linear_model import LogisticRegression

X_train = np.load("../data/subsamp_data/processed_X_train.npy")
X_test = np.load("../data/subsamp_data/processed_X_test.npy")
X_val = np.load("../data/subsamp_data/processed_X_val.npy")
y_train = np.load("../data/subsamp_data/processed_y_train.npy")
y_test = np.load("../data/subsamp_data/processed_y_test.npy")
y_val = np.load("../data/subsamp_data/processed_y_val.npy")

# X_train = np.load("./data/origin_data/X_train.npy")
# X_test = np.load("./data/origin_data/X_test.npy")
# X_val = np.load("./data/origin_data/X_val.npy")
# y_train = np.load("./data/origin_data/y_train.npy")
# y_test = np.load("./data/origin_data/y_test.npy")
# y_val = np.load("./data/origin_data/y_val.npy")

clf = LogisticRegression(C=100, class_weight={0: 0.1, 1: 0.9}).fit(X_train, y_train)
coef = clf.coef_[0]
intercept = clf.intercept_ [0]

np.save("model_LR/w", clf.coef_[0])
np.save("model_LR/b", clf.intercept_[0])
print(coef, '\n', intercept)

# print('Accuracy of Logistic regression classifier on training set: {:.2f}'
#      .format(clf.score(X_train, y_train)))
# print('Accuracy of Logistic regression classifier on validation set: {:.2f}'
#      .format(clf.score(X_val, y_val)))
# print('Accuracy of Logistic regression classifier on test set: {:.2f}'
#      .format(clf.score(X_test, y_test)))

val_label_predict = clf.predict(X_val)

TP = 0
FP = 0
TN = 0
FN = 0

for index in range(len(y_val)):

    # fraud transaction
    if y_val[index] == 1:

        if val_label_predict[index] == 1:
            TP += 1
        else:
            FN += 1

    # normal transaction
    else:
        if val_label_predict[index] == 0:
            TN += 1
        else:
            FP += 1

print('TP count:    {}'.format(TP))
print('FP count:    {}'.format(FP))
print('TN count:    {}'.format(TN))
print('FN count:    {}'.format(FN))

print('Precision rate:  {}'.format(TP/(TP+FP)))
print('Recall rate: {}'.format(TP/(TP+FN)))

print('Mean score:  {}'.format(clf.score(X_val, y_val)))
