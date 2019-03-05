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

def remap(value):
    value -= 63/2
    return value

def segmentize(lst, segmentSize=0, overlapSize=0):
    if segmentSize == 0:
        return [lst,]
    
    if overlapSize >= segmentSize:
        raise Exception('Overlap size > segment size.')

    result = []
    for i in range(0, len(lst), segmentSize - overlapSize):
        if i + segmentSize > len(lst):
            break
        result.append(lst[i:i+segmentSize])
    return result

def dtw_segmentize(x_axis, y_axis, z_axis, threshold, window_size):
    total_acc = np.array(x_axis) + np.array(y_axis) + np.array(z_axis)

    segmented_x_axis = []
    segmented_y_axis = []
    segmented_z_axis = []

    prev = total_acc[0:window_size]
    start = 0
    end = window_size

    for i in range(window_size, len(total_acc) - window_size, window_size): 
        curr = total_acc[i:i+window_size]
        dist, mappings = dtw(prev, curr)
        
        if dist/len(mappings) >= threshold:
            segmented_x_axis.append(x_axis[start:end])
            segmented_y_axis.append(y_axis[start:end])
            segmented_z_axis.append(z_axis[start:end])

            start = i
            end = i + window_size
        else:
            end += window_size

        prev = curr

    segmented_x_axis.append(x_axis[start:end])
    segmented_y_axis.append(y_axis[start:end])
    segmented_z_axis.append(z_axis[start:end])

    return segmented_x_axis, segmented_y_axis, segmented_z_axis


def combine(dtw_seg=False, dtw_threshold=100, dtw_window_size=150, 
segmentSize=0, overlapSize=0, combinedDF_outfile=None, countsDF_outfile=None, need_remap=False,
use_cache=True):

    if use_cache and combinedDF_outfile and os.path.isfile(combinedDF_outfile) and countsDF_outfile and os.path.isfile(countsDF_outfile):
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
    consolidatedData['overlap_size'] = []

    print('Combining and segmentizing data...', flush=True)

    counts = {}
    for f in tqdm(filenames):
        matcher = extractInfo(f)
        data = pd.read_csv(f, names=columns,sep=r'\s{1,}',engine='python')
        
        if dtw_seg:
            x_axis, y_axis, z_axis = dtw_segmentize(
                data['x axis'], data['y axis'], data['z axis'],
                dtw_threshold, dtw_window_size)
        else:
            x_axis = segmentize(data['x axis'], segmentSize, overlapSize)
            y_axis = segmentize(data['y axis'], segmentSize, overlapSize)
            z_axis = segmentize(data['z axis'], segmentSize, overlapSize)

        if need_remap:
            for axis in [x_axis, y_axis, z_axis]:
                for seg in axis:
                    print(seg[0])
                    for i in range(len(seg)):
                        seg[i] = remap(seg[i])

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

        for i in range(0, len(x_axis)):

            consolidatedData['user'].append(user)
            consolidatedData['activity'].append(activity)

            consolidatedData['year'].append(year)
            consolidatedData['month'].append(month)
            consolidatedData['day'].append(day)

            consolidatedData['HH'].append(hour)
            consolidatedData['MM'].append(minute)
            consolidatedData['SS'].append(second)

            consolidatedData['total_duration'].append(len(data['x axis']))

            consolidatedData['x_axis'].append(np.array(x_axis[i]))
            consolidatedData['y_axis'].append(np.array(y_axis[i]))
            consolidatedData['z_axis'].append(np.array(z_axis[i]))

            consolidatedData['overlap_size'].append(overlapSize)

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
    print('Usage: "./combine.py [segment_size] [overlap_size] [outputfile]"' )
        
    segmentSize = 0 if len(sys.argv) < 2 else int(sys.argv[1])
    overlapSize = 0 if len(sys.argv) < 3 else int(sys.argv[2])
    combinedDF_outfile = None if len(sys.argv) < 4 else sys.argv[3]

    combine(segmentSize, overlapSize, combinedDF_outfile)