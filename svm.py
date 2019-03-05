#!/usr/bin/env python3

import pandas as pd
import numpy as np
from tqdm import tqdm as tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from numpy import nan as NaN

import joblib
import os
import sys

ACTIVITIES = ['Brush_teeth','Climb_stairs','Comb_hair','Descend_stairs','Drink_glass','Eat_meat','Eat_soup','Getup_bed','Liedown_bed','Pour_water','Sitdown_chair','Standup_chair','Use_telephone','Walk']
STATS = ['precision', 'recall', 'support']

def evaluate(dataset, availableFeatures, toBeUsedFeatures, toSaveUserModel=True, outputfile=None):
    flattenedFeatures = []
    for toBeUsedFeature in toBeUsedFeatures:
        flattenedFeatures.extend(availableFeatures[toBeUsedFeature])

    results = {'user': [], 'test_accuracy': [], 'train_accuracy': [], 'features_used': [], 'segment_size': [], 'overlap_size': []}

    print('Classifying and evaluating...', flush=True)
    models = {}

    for user in tqdm(dataset.user.unique()):
        if user in ['f5', 'm10', 'm11']:
            continue

        filteredDataset = dataset[dataset.user == user]
        X = filteredDataset[flattenedFeatures]

        # Checks if the data set contains any NaN
        nullValues = X[X.isnull().values]
        if not nullValues.empty:
            print(nullValues)
            raise Exception('Dataset contains NaN. See above.')

        Y = filteredDataset[['activity']]
        m = np.arange(0.0, 1.0, 0.1)

        # Split the dataset into training and test
        X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.25,random_state=70)

        classifier = SVC(gamma='scale', decision_function_shape='ovo')  

        classifier.fit(X_train.values, y_train.values.ravel())  
        y_pred = classifier.predict(X_test)
        
        results['user'].append(user)
        results['test_accuracy'].append(accuracy_score(y_test,y_pred)*100)
        results['train_accuracy'].append(accuracy_score(y_train, classifier.predict(X_train))*100)
        results['features_used'].append('|'.join(toBeUsedFeatures))
        results['segment_size'].append(len(dataset.iloc[0]['x_axis']))
        results['overlap_size'].append(filteredDataset.iloc[0]['overlap_size'])

        classReport = classification_report(y_test,y_pred,output_dict=True)

        for activity in ACTIVITIES:
            for stat in STATS:
                newHeader = activity.lower() + '_' + stat
                if newHeader not in results:
                    results[newHeader] = []
                results[newHeader].append(classReport[activity.lower()][stat] if activity.lower() in classReport else '')

        if toSaveUserModel:
            os.makedirs('joblib_output', exist_ok=True)
            joblib.dump(classifier, 'joblib_output/' + user + '.joblib')
        models[user] = classifier

    resultDF = pd.DataFrame(results)
    if outputfile:
        if os.path.isfile(outputfile):
            resultDF.to_csv(outputfile, mode='a', header=False, index=False)
        else:
            resultDF.to_csv(outputfile, index=False)

    return resultDF, models

if __name__ == "__main__":
    print('Usage: "./evaluate.py [data_pickle] [outputfile]"' )
    if len(sys.argv) < 2:
        raise FileNotFoundError 
    else:
        data_pickle = sys.argv[1]
        dataset = pd.read_pickle(data_pickle)
        add_features(dataset, True, 'random_forest.csv')
    
