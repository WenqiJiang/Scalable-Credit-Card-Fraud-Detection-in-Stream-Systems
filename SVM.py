import numpy as np
from sklearn.svm import SVC

# load features and label from training set into array
train_features = np.load("./data/subsamp_data/processed_X_train.npy")
train_label = np.load("./data/subsamp_data/processed_y_train.npy")

# load features and label from validation set into array
val_features = np.load("./data/subsamp_data/processed_X_val.npy")
val_label = np.load("./data/subsamp_data/processed_y_val.npy")

svc = SVC(C=10).fit(train_features,train_label)        

val_label_predict = svc.predict(val_features)

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

print('Mean score:  {}'.format(svc.score(val_features,val_label)))