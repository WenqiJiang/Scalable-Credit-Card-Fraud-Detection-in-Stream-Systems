import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler


# credit_card = np.genfromtxt('C:/Users/lming/OneDrive/Desktop/creditcard.csv', dtype='str', delimiter=',', skip_header=1)
# credit_card = np.char.strip(credit_card, '"').astype(float)
#
# X_fruits = credit_card[:, :-1]
# y_fruits = credit_card[:, -1]
# y_fruits = np.array(y_fruits, dtype=np.int32)
#
# X_train, X_test, y_train, y_test = train_test_split(X_fruits, y_fruits, test_size=0.2, random_state=1)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
# np.save("./data/X_train", X_train)
# np.save("./data/y_train", y_train)
# np.save("./data/X_test", X_test)
# np.save("./data/y_test", X_test)
# np.save("./data/X_val", X_val)
# np.save("./data/y_val", y_val)
X_train = np.load("./data/subsamp_data/processed_X_train.npy")
X_test = np.load("./data/subsamp_data/processed_X_test.npy")
X_val = np.load("./data/subsamp_data/processed_X_val.npy")
y_train = np.load("./data/subsamp_data/processed_y_train.npy")
y_test = np.load("./data/subsamp_data/processed_y_test.npy")
y_val = np.load("./data/subsamp_data/processed_y_val.npy")

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train_scaled, y_train)
# print('Accuracy of K-NN classifier on training set: {:.2f}'
#      .format(knn.score(X_train_scaled, y_train)))
# print('Accuracy of K-NN classifier on validation set: {:.2f}'
#      .format(knn.score(X_val_scaled, y_val)))
# print('Accuracy of K-NN classifier on test set: {:.2f}'
#      .format(knn.score(X_test_scaled, y_test)))

val_label_predict = knn.predict(X_val_scaled)

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

print('Mean score:  {}'.format(knn.score(X_val_scaled,y_val)))