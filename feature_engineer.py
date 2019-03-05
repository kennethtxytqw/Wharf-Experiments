#!/usr/bin/env python3
 
import pandas as pd
import math
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from evaluate import ACTIVITIES
import sys
import pickle
import os

def extractFeatures(x_axis, y_axis, z_axis, user):

    features = {}

    features['x_mean'] = x_axis.mean()
    features['y_mean'] = y_axis.mean()
    features['z_mean'] = z_axis.mean()

    features['x_std'] = x_axis.std()
    features['y_std'] = y_axis.std()
    features['z_std'] = z_axis.std()
    
    features['xy_corr'] = np.correlate(x_axis,y_axis)
    features['xz_corr'] = np.correlate(x_axis,z_axis)
    features['yz_corr'] = np.correlate(y_axis,z_axis)

    features['x_freq'] = np.abs(np.fft.rfft(x_axis))**2
    features['y_freq'] = np.abs(np.fft.rfft(y_axis))**2
    features['z_freq'] = np.abs(np.fft.rfft(z_axis))**2

    features['x_energy']  = sum(features['x_freq'])/len(features['x_freq'])
    features['y_energy']  = sum(features['y_freq'])/len(features['y_freq'])
    features['z_energy']  = sum(features['z_freq'])/len(features['z_freq'])

    features['x_entropy'] = entropy(features['x_freq']/sum(features['x_freq']))
    features['y_entropy'] = entropy(features['y_freq']/sum(features['y_freq']))
    features['z_entropy'] = entropy(features['z_freq']/sum(features['z_freq']))

    features['pitch_mean'] = np.arctan(features['x_mean']/math.sqrt(np.abs(features['y_mean'] + features['z_mean'])))
    features['roll_mean'] = np.arctan(features['y_mean']/math.sqrt(np.abs(features['x_mean'] + features['z_mean'])))
    features['yaw_mean'] = np.arctan(features['z_mean']/math.sqrt(np.abs(features['y_mean'] + features['x_mean'])))

    # for activity in ACTIVITIES:
    #     if len(centroids[user][activity.lower()]['x_axis']) != 0:
    #         features[activity + 'x_dtw_dist'], path = fastdtw(centroids[user][activity.lower()]['x_axis'], x_axis)
    #         features[activity + 'y_dtw_dist'], path = fastdtw(centroids[user][activity.lower()]['y_axis'], y_axis)
    #         features[activity + 'z_dtw_dist'], path = fastdtw(centroids[user][activity.lower()]['z_axis'], z_axis)
    #     else:
    #         features[activity + 'x_dtw_dist'] = sys.maxsize
    #         features[activity + 'y_dtw_dist'] = sys.maxsize
    #         features[activity + 'z_dtw_dist'] = sys.maxsize

    return features

def extractCentroid(dataset):

    print('Extracting centroids...', flush=True)
    centroids = {}
    for user in tqdm(dataset.user.unique()):
        for activity in dataset.activity.unique():
            
            num = len(dataset[(dataset.user == user) & (dataset.activity == activity)])
            
            
            if user not in centroids:
                centroids[user] = {}

            centroids[user][activity] = {}
            

            if num == 0:
                centroids[user][activity]['x_axis'] = []
                centroids[user][activity]['y_axis'] = []
                centroids[user][activity]['z_axis'] = []
                continue

            best_x_axis = 0
            best_y_axis = 0
            best_z_axis = 0

            best_squared_dist = sys.maxsize

            for row1 in dataset[(dataset.user == user) & (dataset.activity == activity)].itertuples():
                best_x_axis = np.add(best_x_axis, row1.x_axis)
                best_y_axis = np.add(best_y_axis, row1.y_axis)
                best_z_axis = np.add(best_z_axis, row1.z_axis)
            
            centroids[user][activity]['x_axis'] = np.divide(best_x_axis,num)
            centroids[user][activity]['y_axis'] = np.divide(best_y_axis,num)
            centroids[user][activity]['z_axis'] = np.divide(best_z_axis,num)

    return centroids


def add_features(dataset, outputfile=None, use_cache=True):
    if use_cache and outputfile and os.path.isfile(outputfile) and os.path.isfile('_' + outputfile):
        dataset = pd.read_pickle(outputfile)
        availableFeatures = pickle.load('_' + outputfile)
        return dataset, availableFeatures

    availableFeatures = {
        'acc_means': ['x_mean', 'y_mean', 'z_mean'],
        'acc_corrs': ['xy_corr', 'xz_corr', 'yz_corr'],
        'acc_stds': ['x_std', 'y_std', 'z_std'], 
        'energies': ['x_energy', 'y_energy', 'z_energy'],
        'entropies': ['x_entropy', 'y_entropy', 'z_entropy'],
        'time': ['HH', 'total_duration'],
        'rotation_means': ['pitch_mean', 'yaw_mean', 'roll_mean'],
        # 'dtw_dist': [],
    }

    consolidatedFeatures = {}

    # for activity in ACTIVITIES:
    #     availableFeatures['dtw_dist'].append(activity + 'x_dtw_dist')
    #     availableFeatures['dtw_dist'].append(activity + 'y_dtw_dist')
    #     availableFeatures['dtw_dist'].append(activity + 'z_dtw_dist')

    print('Extracting features...', flush=True)
    for row in tqdm(dataset.itertuples()):
        
        features = extractFeatures(row.x_axis, row.y_axis, row.z_axis, row.user)

        for feature, value in features.items():
            if feature not in consolidatedFeatures:
                consolidatedFeatures[feature] = []
            consolidatedFeatures[feature].append(value)

    for feature, values in consolidatedFeatures.items():
        dataset[feature] = values

    if outputfile:
        dataset.to_pickle(outputfile)
        pickle.dump(availableFeatures, open('_' + outputfile, 'wb'))
    return dataset, availableFeatures

if __name__ == "__main__":
    print('Usage: "./feature_engineer.py [data_pickle] [outputfile]"')
    if len(sys.argv) < 2:
        raise FileNotFoundError 
    else:
        data_pickle = sys.argv[1]
        dataset = pd.read_pickle(data_pickle)
        add_features(dataset)