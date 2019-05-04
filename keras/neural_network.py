import numpy as np
import argparse

from keras.models import Sequential
from keras.layers import Dense
from prec_recall import prec_recall
from prec_recall import prec_recall

# fix random seed for reproducibility
np.random.seed(7)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="origin", help="origin / subsamp")
parser.add_argument('--fraud_weight', type=float, default=1, help="usually larger than 1")
parser.add_argument('--epochs', type=int, default=10, help="epoch number")
args = parser.parse_args()
dataset = args.dataset
fraud_weight = args.fraud_weight
epochs = args.epochs

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

model = Sequential()
model.add(Dense(12, input_dim=28, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

class_weight = {0: 1,
                1: fraud_weight}
model.fit(X_train, y_train, epochs=epochs, batch_size=10, class_weight=class_weight)

model.save_weights("models/model.h5")
my_pred = model.predict(X_val)
prec_recall(my_pred, y_val)