import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler



credit_card = np.genfromtxt('C:/Users/lming/OneDrive/Desktop/creditcard.csv', dtype='str', delimiter=',', skip_header=1)
credit_card = np.char.strip(credit_card, '"').astype(float)
#for i in range(20):
    #print(credit_card[i])

X_fruits = credit_card[:, :-1]
y_fruits = credit_card[:, -1]
y_fruits = np.array(y_fruits, dtype=np.int32)
for i in range(20):
    print(y_fruits[i])
X_train, X_test, y_train, y_test = train_test_split(X_fruits, y_fruits, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
# np.save("./data/X_train", X_train)
# np.save("./data/y_train", y_train)
# np.save("./data/X_test", X_test)
# np.save("./data/y_test", X_test)
# np.save("./data/X_val", X_val)
# np.save("./data/y_val", y_val)
# scaler = MinMaxScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# # K-Nearest Neighbors
# knn = KNeighborsClassifier(n_neighbors = 5)
# knn.fit(X_train_scaled, y_train)
# print('Accuracy of K-NN classifier on training set: {:.2f}'
#      .format(knn.score(X_train_scaled, y_train)))
# print('Accuracy of K-NN classifier on test set: {:.2f}'
#      .format(knn.score(X_test_scaled, y_test)))
