#!/usr/bin/env python3

import numpy as np
import os
from Utilities import FileUtilities
from matplotlib import pyplot as plt
from Utilities import EarthMovingDistance

def main():
	path = '/Users/Apple/Documents/gitRepo/ra-eeyes-activity-detection/data_input/'
	utl = FileUtilities(path)
	amplitude_matrices = utl.get_amplitude_matrices()
	print("Number of matrices read = ", len(amplitude_matrices))

	histograms = []
	for amplitude_matrix in amplitude_matrices:
		for i in amplitude_matrix:
			freq = np.zeros(40)
			for j in i:
				freq[int(j)] += 1
		histograms.append(freq)

	labels_np = np.array(utl.labels)
	print(utl.labels)
	emd = EarthMovingDistance(histograms)
	emd_matrix = emd.get_EMD_matrix()
	closest_activity = emd.get_closest_activity()
	predicted = labels_np[closest_activity]
	print("______________________ EMD matrix start ______________________")
	print(emd_matrix)
	print("______________________ EMD matrix ends _______________________")
	print("___________________ closest_activity start ___________________")
	print(predicted)
	print("___________________ closest_activity ends ____________________")

if __name__ == '__main__':
	main()