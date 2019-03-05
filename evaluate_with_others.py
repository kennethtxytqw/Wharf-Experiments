#!/usr/bin/env python3

import pandas as pd
import joblib
from tqdm import tqdm as tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from evaluate import ACTIVITIES, STATS

import os
import sys

def evaluate_with_others(classifier, dataset, availableFeatures, toBeUsedFeatures, outputfile=None):

    flattenedFeatures = []
    for toBeUsedFeature in toBeUsedFeatures:
        flattenedFeatures.extend(availableFeatures[toBeUsedFeature])

    results = {'user': [], 'test_accuracy': [], 'features_used': [], 'segment_size': [], 'overlap_size': []}
    for user in tqdm(dataset.user.unique()):

        filteredDataset = dataset[dataset.user == user]
        X = filteredDataset[flattenedFeatures]

        # Checks if the data set contains any NaN
        nullValues = X[X.isnull().values]
        if not nullValues.empty:
            print(nullValues)
            raise Exception('Dataset contains NaN. See above.')

        Y = filteredDataset[['activity']]

        y_pred = classifier.predict(X)
        
        results['user'].append(user)
        results['test_accuracy'].append(accuracy_score(Y,y_pred)*100)
        results['features_used'].append('|'.join(toBeUsedFeatures))
        results['segment_size'].append(len(dataset.iloc[0]['x_axis']))
        results['overlap_size'].append(filteredDataset.iloc[0]['overlap_size'])

        classReport = classification_report(Y,y_pred,output_dict=True)

        for activity in ACTIVITIES:
            for stat in STATS:
                newHeader = activity.lower() + '_' + stat
                if newHeader not in results:
                    results[newHeader] = []
                results[newHeader].append(classReport[activity.lower()][stat] if activity.lower() in classReport else '')
        
    resultDF = pd.DataFrame(results)
    if outputfile:
        if os.path.isfile(outputfile):
            resultDF.to_csv(outputfile, mode='a', header=False, index=False)
        else:
            resultDF.to_csv(outputfile, index=False)

    return resultDF

if __name__ == "__main__":
    print('Usage: "./evaluateWithOthers.py [model_joblib] [data_pickle] [outputfile]"' )
    if len(sys.argv) < 2:
        raise FileNotFoundError 
    else:
        model = joblib.load(sys.argv[1])
        dataset = pd.read_pickle(data_pickle)
        evaluate_with_others(model, dataset, True, 'random_forest.csv')