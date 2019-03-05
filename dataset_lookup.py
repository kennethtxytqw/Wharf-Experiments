#!/usr/bin/env python3

import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from fastdtw import dtw
import re
import sys
import os

FILENAME_REGEX = r"WHARF Data Set/Data/.+/Accelerometer-(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})-(?P<HH>\d{2})-(?P<MM>\d{2})-(?P<SS>\d{2})-(?P<activity>[^-]+)-(?P<user>\w\d+).txt"
FILENAME_PATTERN = re.compile(FILENAME_REGEX)

def extractInfo(filename):
    matcher = FILENAME_PATTERN.match(filename)
    if not matcher:
        raise Exception("Invalid filename convention: " + filename)
    else:
        return matcher

def combine(combinedDF_outfile=None, countsDF_outfile=None):

    if combinedDF_outfile and os.path.isfile(combinedDF_outfile) and countsDF_outfile and os.path.isfile(countsDF_outfile):
        combinedDF = pd.read_pickle(combinedDF_outfile)
        countsDF = pd.read_pickle(countsDF_outfile)
        return combinedDF, countsDF

    filenames =[]
    for name in glob.glob('WHARF Data Set/Data/*/*.txt'):
        filenames.append(name)

    columns = ['x axis','y axis','z axis']

    consolidatedData = {}

    consolidatedData['user'] = []
    consolidatedData['activity'] = []
    consolidatedData['year'] = []
    consolidatedData['month'] = []
    consolidatedData['day'] = []
    consolidatedData['HH'] = []
    consolidatedData['MM'] = []
    consolidatedData['SS'] = []
    consolidatedData['total_duration'] = []
    consolidatedData['x_axis'] = []
    consolidatedData['y_axis'] = []
    consolidatedData['z_axis'] = []

    print('Loading data...', flush=True)

    counts = {}
    for f in tqdm(filenames):
        matcher = extractInfo(f)
        data = pd.read_csv(f, names=columns,sep=r'\s{1,}',engine='python')

        x_axis = data['x axis']
        y_axis = data['y axis']
        z_axis = data['z axis']

        user = matcher.group("user")
        activity = matcher.group('activity')
        year = matcher.group('year')
        month = matcher.group('month')
        day = matcher.group('day')
        hour = matcher.group('HH')
        minute = matcher.group('MM')
        second = matcher.group('SS')

        if user not in counts:
            counts[user] = {}
        if activity not in counts[user]:
            counts[user][activity] = 0
        counts[user][activity] = counts[user][activity] + len(data['x axis'])

        consolidatedData['user'].append(user)
        consolidatedData['activity'].append(activity)

        consolidatedData['year'].append(year)
        consolidatedData['month'].append(month)
        consolidatedData['day'].append(day)

        consolidatedData['HH'].append(hour)
        consolidatedData['MM'].append(minute)
        consolidatedData['SS'].append(second)

        consolidatedData['total_duration'].append(len(data['x axis']))

        consolidatedData['x_axis'].append(np.array(x_axis))
        consolidatedData['y_axis'].append(np.array(y_axis))
        consolidatedData['z_axis'].append(np.array(z_axis))

    combinedDF = pd.DataFrame(data=consolidatedData)
    
    print('Aggregating activity sample counts...', flush=True)
    countsDF = {'user':[], 'activity':[], 'count':[]}
    for user in tqdm(counts.keys()):
        for activity in counts[user].keys():
            countsDF['count'].append(counts[user][activity])
            countsDF['user'].append(user)
            countsDF['activity'].append(activity)
    countsDF = pd.DataFrame(countsDF)
    
    if combinedDF_outfile and countsDF_outfile:
        combinedDF.to_pickle(combinedDF_outfile)
        countsDF.to_pickle(countsDF_outfile)

    return combinedDF, countsDF

if __name__ == "__main__":
    print('Usage: "./dataset_lookup.py [outputfile]"' )

    combinedDF_outfile = None if len(sys.argv) < 1 else sys.argv[1]

    combine(combinedDF_outfile, 'counts.pkl')