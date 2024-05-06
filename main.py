import time
import json
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from typing import List
import random


#The function gets the training set, the validation set, the radius and the distances from the validation set to the training set and returns the predicted label
def classifier(trn_features, trn_labels, vld_features, radius, distances):
    predicted_labels = []  

    # Iterate over the distances with corresponding indices
    for i, distance in enumerate(distances):
        if distance <= radius:  
            predicted_labels.append(trn_labels[i])  # Add the corresponding label to list of predicted labels

    if len(predicted_labels) == 0:  
        return random.choice(trn_labels)  # Return a random label from the training set if no label was found

    predicted_labels.sort()  
    label_that_appear_the_most = max(set(predicted_labels), key=predicted_labels.count)  # find the label that appears the most in the list of predicted labels

    return label_that_appear_the_most 


def training(df_trn: DataFrame, df_vld: DataFrame, num_iterations: int) -> float:
    scaler = StandardScaler()
    best_radius = 0
    best_accuracy = 0

    # get the features and labels from the data frames as numpy arrays
    train_features = df_trn.iloc[:, :-1].values
    train_labels = df_trn.iloc[:, -1].values
    valid_features = df_vld.iloc[:, :-1].values
    valid_labels = df_vld.iloc[:, -1].values

    # normalize the features with the scaler and fit the scaler to the training set
    train_features_with_scaling = scaler.fit_transform(train_features)
    valid_features_with_scaling = scaler.fit_transform(valid_features)

    distances = euclidean_distances(valid_features_with_scaling, train_features_with_scaling)

    # calculate the average distance and the average minimal distance
    average_distance = np.mean(distances)
    average_minimal_distance = np.mean(np.min(distances, axis=1))
    diff = average_distance - average_minimal_distance # calculate the difference between the two

    # iterate over the number of iterations and calculate the radius for each iteration
    for itr in range(1, num_iterations + 1):
        labels_predicted = []

        for i in range(len(df_vld)):
            # get the features of each row in the validation set
            vld_single_row_features = df_vld.iloc[i, :-1].values

            label_predicted = classifier(train_features_with_scaling, train_labels, vld_single_row_features, (diff / num_iterations) * itr, distances[i]) # get the predicted label for each row in the validation set
            labels_predicted.append(label_predicted)

        accuracy = sum(1 for x, y in zip(labels_predicted, valid_labels) if x == y) / len(labels_predicted)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_radius = (diff / num_iterations) * itr

    return best_radius

def prediction(df_tst: DataFrame, train_features_with_scaling, train_labels, best_radius) -> List:
    predictions = []

    scaler = StandardScaler()
    test_features = df_tst.iloc[:, :].values
    test_features_with_scaling = scaler.fit_transform(test_features)

    # calculate the distances between the test set and the training set and save in a matrix
    distances = euclidean_distances(test_features_with_scaling, train_features_with_scaling)

    for i in range(len(df_tst)):
        # get the features of each row in the test set
        test_features_single_row = df_tst.iloc[i, :].values

        # get the predicted label for each row in the test set
        label_predicted = classifier(train_features_with_scaling, train_labels, test_features_single_row, best_radius, distances[i])


        predictions.append(label_predicted)

    return predictions

def classify_with_NNR(data_trn: str, data_vld: str, df_tst: DataFrame) -> List:
    #set the number of iterations of the algorithm
    num_iterations = 30

    #read the data from the csv files
    df_trn = pd.read_csv(data_trn)
    df_vld = pd.read_csv(data_vld)
    tst_features = df_tst.iloc[:, :].values # get the features from the test set

    print(f'starting classification with {data_trn}, {data_vld}, predicting on {len(df_tst)} instances')

    best_radius = training(df_trn, df_vld, num_iterations) # get the best radius from the training set and the validation set

    scaler = StandardScaler() # create a scaler

    # get the features and labels from the training set
    train_labels = df_trn.iloc[:, -1].values
    train_features = df_trn.iloc[:, :-1].values
    
    train_features_with_scaling = scaler.fit_transform(train_features) # normalize the training set

    predictions = prediction(df_tst, train_features_with_scaling, train_labels, best_radius) # get the predictions for the test set

    return predictions



if __name__ == '__main__':
    start = time.time()

    with open('config.json', 'r', encoding='utf8') as json_file:
        config = json.load(json_file)

    df = pd.read_csv(config['data_file_test'])
    predicted = classify_with_NNR(config['data_file_train'],
                                  config['data_file_validation'],
                                  df.drop(['class'], axis=1))

    labels = df['class'].values
    if not predicted:  # empty prediction, should not happen in your implementation
        predicted = list(range(len(labels)))

    assert(len(labels) == len(predicted))  # make sure you predict label for all test instances
    print(f'test set classification accuracy: {accuracy_score(labels, predicted)}')

    print(f'total time: {round(time.time()-start, 0)} sec')
