#!/usr/bin/env python3

import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from fastdtw import dtw
import re
import sys

FILENAME_REGEX = r"WHARF Data Set/Data/.+/Accelerometer-(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})-(?P<HH>\d{2})-(?P<MM>\d{2})-(?P<SS>\d{2})-(?P<activity>[^-]+)-(?P<user>\w\d+).txt"
FILENAME_PATTERN = re.compile(FILENAME_REGEX)

def extractInfo(filename):
    matcher = FILENAME_PATTERN.match(filename)
    if not matcher:
        raise Exception("Invalid filename convention: " + filename)
    else:
        return matcher

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

x_axes = []
y_axes = []
z_axes = []

combined_axes = []
avg_axes = []


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

    if user == 'f1' and activity.lower() == 'brush_teeth':
        x_axes.append(x_axis)
        y_axes.append(y_axis)
        z_axes.append(z_axis)

        combined_axes.append(np.add(np.add(np.absolute(np.array(x_axis)), np.absolute(np.array(y_axis))), np.absolute(np.array(z_axis))))
        avg_axes.append(np.add(np.add(np.array(x_axis), np.array(y_axis)), np.array(z_axis)))



# print('x_axes dtw values')
# for i in range(len(x_axes)-1):
#     for j in range(i, len(x_axes)):
#         dist, mapping = dtw(x_axes[i], x_axes[j])
#         print(dist)

# print('y_axes dtw values')
# for i in range(len(y_axes)-1):
#     for j in range(i, len(y_axes)):
#         dist, mapping = dtw(y_axes[i], y_axes[j])
#         print(dist)

# print('z_axes dtw values')
# for i in range(len(z_axes)-1):
#     for j in range(i, len(z_axes)):
#         dist, mapping = dtw(z_axes[i], z_axes[j])
#         print(dist)

# print('combined_axes dtw values')
# for i in range(len(combined_axes)-1):
#     for j in range(i, len(combined_axes)):

#         min_len = min(len(combined_axes[i]), len(combined_axes[j]))
#         dist, mapping = dtw(combined_axes[i][:min_len], combined_axes[j][:min_len])
#         print(dist)


# print('avg_axes dtw values')
# for i in range(len(avg_axes)-1):
#     for j in range(i, len(avg_axes)):

#         min_len = min(len(combined_axes[i]), len(combined_axes[j]))
#         dist, mapping = dtw(avg_axes[i][100:min_len], avg_axes[j][100:min_len])
#         print(dist)

window_size = 32
for i in range(0, len(avg_axes[0])-window_size, window_size):
    dist, mappings = dtw(avg_axes[0][i:i+window_size], avg_axes[0][i+window_size:i+window_size*2])
    print(dist/len(mappings))

