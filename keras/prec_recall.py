def prec_recall(label, predict):
    """
    label: real lable, a 1-d array
    predict: predicted lable, a 1-d array
    """

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for index in range(len(label)):

        # fraud transaction
        if label[index] == 1:

            if predict[index] == 1:
                TP += 1
            else:
                FN += 1

        # normal transaction  
        else:
            if predict[index] == 0:
                TN += 1
            else:
                FP += 1

    print('TP count:    {}'.format(TP))
    print('FP count:    {}'.format(FP))
    print('TN count:    {}'.format(TN))
    print('FN count:    {}'.format(FN))

    print('Precision rate:  {}'.format(TP/(TP+FP)))
    print('Recall rate: {}'.format(TP/(TP+FN)))