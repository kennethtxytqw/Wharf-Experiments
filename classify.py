#!/usr/bin/env python3

import pandas as pd
import numpy as np
from tqdm import tqdm as tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from numpy import nan as NaN
import os

def readData(filePath):
    data = pd.read_pickle(filePath)
    return data

dataset = readData('combined.pkl')

# Split the dataset into training and test
featuresToUse = []
means = ['x_mean', 'y_mean', 'z_mean']
corrs = ['xy_corr', 'xz_corr', 'yz_corr']
stds = ['x_std', 'y_std', 'z_std'] 

featuresToUse.extend(means)
featuresToUse.extend(corrs)
featuresToUse.extend(stds)

results = {'user': [], 'test_accuracy': [], 'train_accuracy': [], 'features_used': []}
for user in dataset.user.unique():

    filteredDataset = dataset[dataset.user == user]
    X = filteredDataset[featuresToUse]

    # Checks if the data set contains any NaN
    nullValues = X[X.isnull().values]
    if not nullValues.empty:
        print(nullValues)
        raise Exception('Dataset contains NaN. See above.')

    Y = filteredDataset[['activity']]
    m = np.arange(0.0, 1.0, 0.1)

    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.25,random_state=70)

    classifier = RandomForestClassifier(n_estimators=37, random_state=50,min_samples_leaf=1,max_features='sqrt')  
    classifier.fit(X_train.values, y_train.values.ravel())  
    y_pred = classifier.predict(X_test)

    # print('X_train Shape:', X_train.shape)
    # print('X_test Shape:', X_test.shape)
    # print('y_train Shape:', y_train.shape)
    # print('y_test Shape:', y_test.shape)

    results['user'].append(user)
    results['test_accuracy'].append(accuracy_score(y_test,y_pred)*100)
    results['train_accuracy'].append(accuracy_score(y_train, classifier.predict(X_train))*100)
    results['features_used'].append('|'.join(featuresToUse))

if os.path.isfile('RandomForestClassifier.csv'):
    pd.DataFrame(results).to_csv('RandomForestClassifier.csv', mode='a', header=False)
else:
    pd.DataFrame(results).to_csv('RandomForestClassifier.csv')
