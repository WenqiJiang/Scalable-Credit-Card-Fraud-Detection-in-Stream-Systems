import numpy as np
from sklearn.ensemble import RandomForestClassifier

# load features and label from training set into array


use_origin = False

# if use_origin = True, use original dataset
if use_origin:

    train_features = np.load("../data/origin_data/X_train.npy")
    train_label = np.load("../data/origin_data/y_train.npy")

    val_features = np.load("../data/origin_data/X_val.npy")
    val_label = np.load("../data/origin_data/y_val.npy")



# if use_origin = False, use refined dataset
else:

    train_features = np.load("../data/subsamp_data/processed_X_train.npy")
    train_label = np.load("../data/subsamp_data/processed_y_train.npy")

    val_features = np.load("../data/subsamp_data/processed_X_val.npy")
    val_label = np.load("../data/subsamp_data/processed_y_val.npy")


clf = RandomForestClassifier(n_estimators=100, max_depth=10,random_state=0,class_weight = {0:1,1:100})

clf.fit(train_features,train_label)

val_label_predict = clf.predict(val_features)

TP = 0
FP = 0
TN = 0
FN = 0

for index in range(len(val_label)):

    # fraud transaction
    if val_label[index] == 1:

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

print('Mean score:  {}'.format(clf.score(val_features,val_label)))
