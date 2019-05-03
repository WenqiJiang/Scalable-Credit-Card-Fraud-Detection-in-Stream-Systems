import numpy as np
import operator

def euclidean_distance(vector1, vector2):
    return np.sqrt(np.sum(np.power(vector1-vector2, 2)))


def get_neighbours(X_train, X_test_instance, k):
    distances = []
    neighbors = []
    for i in range(X_train.shape[0]):
        dist = euclidean_distance(X_train[i], X_test_instance)
        distances.append((i, dist))
    distances.sort(key=operator.itemgetter(1))
    for x in range(k):
        #print distances[x]
        neighbors.append(distances[x][0])
    return neighbors

def predictkNNClass(neighbors, y_train):
    classVotes = {}
    for i in range(len(neighbors)):
        if y_train[neighbors[i]] in classVotes:
            classVotes[y_train[neighbors[i]]] += 1
        else:
            classVotes[y_train[neighbors[i]]] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def kNN_test(X_train, X_test, Y_train, Y_test, k):
    output_classes = []
    for i in range(X_test.shape[0]):
        output = get_neighbours(X_train, X_test[i], k)
        predictedClass = predictkNNClass(output, Y_train)
        output_classes.append(predictedClass)
    return output_classes

def prediction_accuracy(predicted_labels, original_labels):
    count = 0
    for i in range(len(predicted_labels)):
        if predicted_labels[i] == original_labels[i]:
            count += 1
    return float(count)/len(predicted_labels)


k = 5
X_train = np.load("../data/subsamp_data/processed_X_train.npy")
X_test = np.load("../data/subsamp_data/processed_X_test.npy")
X_val = np.load("../data/subsamp_data/processed_X_val.npy")
y_train = np.load("../data/subsamp_data/processed_y_train.npy")
y_test = np.load("../data/subsamp_data/processed_y_test.npy")
y_val = np.load("../data/subsamp_data/processed_y_val.npy")

predicted_classes = kNN_test(X_train, X_val, y_train, y_val, k)
final_accuracies = prediction_accuracy(predicted_classes, y_val)
print(final_accuracies)
