import numpy as np
import pandas as pd
import time
import argparse 

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from prec_recall import prec_recall

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default="origin", help="origin / subsamp")
args = parser.parse_args()
dataset = args.dataset

if dataset == "subsample":
    X_train = np.load("../data/subsamp_data/processed_X_train.npy")
    y_train = np.load("../data/subsamp_data/processed_y_train.npy")
elif dataset == "origin":
    X_train = np.load("../data/origin_data/X_train.npy")
    y_train = np.load("../data/origin_data/y_train.npy")
else:
    raise Exception("Unknown dataset name")

# val / test always on the largest dataset
X_val = np.load("../data/origin_data/X_val.npy")
y_val = np.load("../data/origin_data/y_val.npy")
X_test = np.load("../data/origin_data/X_test.npy")
y_test = np.load("../data/origin_data/y_test.npy")

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train_scaled, y_train)

start = time.perf_counter()
val_label_predict = knn.predict(X_val_scaled)
end = time.perf_counter()
print("time:\t{}\nresult:\t{}".format(end - start, val_label_predict[0:10]))

prec_recall(y_val, val_label_predict)
