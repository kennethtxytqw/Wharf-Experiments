#!/usr/bin/env python3

import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import sys

ACTIVITIES = ['Brush_teeth','Climb_stairs','Comb_hair','Descend_stairs','Drink_glass','Eat_meat','Eat_soup','Getup_bed','Liedown_bed','Pour_water','Sitdown_chair','Standup_chair','Use_telephone','Walk']
USERS = ['f1','f2','f3','f4','f5','f6','m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11'] 

FILENAME_REGEX = r"WHARF Data Set/Data/.+/Accelerometer-(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})-(?P<HH>\d{2})-(?P<MM>\d{2})-(?P<SS>\d{2})-(?P<activity>[^-]+)-(?P<user>\w\d+).txt"
FILENAME_PATTERN = re.compile(FILENAME_REGEX)

def extractInfo(filename):
	matcher = FILENAME_PATTERN.match(filename)
	if not matcher:
		raise Exception("Invalid filename convention: " + filename)
	else:
		return matcher

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

def extractFeatures(x_axis, y_axis, z_axis):
	features = {}

	features['x_mean'] = x_axis.mean()
	features['y_mean'] = y_axis.mean()
	features['z_mean'] = z_axis.mean()

	features['x_std'] = x_axis.std()
	features['y_std'] = y_axis.std()
	features['z_std'] = z_axis.std()
	
	features['xy_corr'] = x_axis.corr(y_axis)
	features['xz_corr'] = x_axis.corr(z_axis)
	features['yz_corr'] = y_axis.corr(z_axis)

	return features
	
print('Usage: "./combine.py [segment_size] [overlap_size]"' )

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
consolidatedData['x_axis'] = []
consolidatedData['y_axis'] = []
consolidatedData['z_axis'] = []

segmentSize = 0 if len(sys.argv) < 2 else int(sys.argv[1])
overlapSize = 0 if len(sys.argv) < 3 else int(sys.argv[2])

print('Combining and segmentizing data...')
for f in tqdm(filenames):
	matcher = extractInfo(f)
	data = pd.read_csv(f, names=columns,sep=r'\s{1,}',engine='python')

	x_axis = segmentize(data['x axis'], segmentSize, overlapSize)
	y_axis = segmentize(data['y axis'], segmentSize, overlapSize)
	z_axis = segmentize(data['z axis'], segmentSize, overlapSize)

	user = matcher.group("user")
	activity = matcher.group('activity')
	year = matcher.group('year')
	month = matcher.group('month')
	day = matcher.group('day')
	hour = matcher.group('HH')
	minute = matcher.group('MM')
	second = matcher.group('SS')

	for i in range(0, len(x_axis)):

		consolidatedData['user'].append(user)
		consolidatedData['activity'].append(activity)

		consolidatedData['year'].append(year)
		consolidatedData['month'].append(month)
		consolidatedData['day'].append(day)

		consolidatedData['HH'].append(hour)
		consolidatedData['MM'].append(minute)
		consolidatedData['SS'].append(second)

		consolidatedData['x_axis'].append(np.array(x_axis[i]))
		consolidatedData['y_axis'].append(np.array(y_axis[i]))
		consolidatedData['z_axis'].append(np.array(z_axis[i]))

		features = extractFeatures(x_axis[i], y_axis[i], z_axis[i])

		for feature, value in features.items():
			if feature not in consolidatedData:
				consolidatedData[feature] = []
			consolidatedData[feature].append(value)

data_frame = pd.DataFrame(data=consolidatedData)
data_frame.to_pickle('combined.pkl')

print('Shape of data frame: ' + str(data_frame.shape))
