import numpy as np
import time
import argparse

from sklearn.ensemble import RandomForestClassifier

# load features and label from training set into array


from prec_recall import prec_recall

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default="origin", help="origin / subsamp")
parser.add_argument('--fraud_weight', type=float, default=1, help="usually larger than 1")
parser.add_argument('--n_estimators', type=int, default=100, help="int larger than 1")
parser.add_argument('--max_depth', type=int, default=10, help="int larger than 1")

args = parser.parse_args()
dataset = args.dataset
fraud_weight = args.fraud_weight
n_estimators = args.n_estimators
max_depth = args.max_depth

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

clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,random_state=0,class_weight = {0:1,1:fraud_weight})

clf.fit(X_train, y_train)

start = time.perf_counter()
val_label_predict = clf.predict(X_val)
end = time.perf_counter()

print("time consumed:", end - start)
print("Valiadation:")
prec_recall(y_val, val_label_predict)
print("Test:")
test_label_predict = clf.predict(X_test)
prec_recall(y_test, test_label_predict)
