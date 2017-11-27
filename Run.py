#!/usr/bin/env python3

from CoarseActivityDetection import CoarseActivityDetection

import numpy as np
import os
from Utilities import FileUtilities
# from Utilities import MovingVariance
from matplotlib import pyplot as plt
# from InPlace import InPlace
from Utilities import EarthMovingDistance

# utl = FileUtilities('data1/')

# path = '/Users/Apple/Documents/gitRepo/ra-eeyes-activity-detection/data_input/'
# input_file_sitdown = 'input_161219_siamak_sitdown_1.dat.csv'
# df_sitdown = utl.read_csv(path+input_file_sitdown)
# data_sitdown = utl.get_data_matrix(df_sitdown)
# C_sitdown = data_sitdown[:,1:91]
# mv_sitdown = MovingVariance(C_sitdown,1000)
# V_sitdown = mv_sitdown.get_moving_var()
# print(V_sitdown.shape)
# print(type(V_sitdown))
# CMV_sitdown = mv_sitdown.get_cumulative_moving_variance(V_sitdown)
# plt.plot(range(1, CMV_sitdown.shape[0]+1) , CMV_sitdown)

# input_file_run = 'input_161219_siamak_run_1.dat.csv'
# df_run = utl.read_csv(path+input_file_run)
# data_run = utl.get_data_matrix(df_run)
# df_run = utl.read_csv(input_file_run)
# C_run = data_run[:,1:91]
# mv_run = MovingVariance(C_run,1000)
# V_run = mv_run.get_moving_var()
# # print(V_run.shape)
# # print(type(V_run))
# CMV_run = mv_run.get_cumulative_moving_variance(V_run)
# plt.plot(range(1, CMV_run.shape[0]+1) , CMV_run)

# plt.show()

##################

# path = '/Users/Apple/Documents/gitRepo/ra-eeyes-activity-detection/data_small/'
# files = os.listdir(path)
# utl = FileUtilities(path)
# data_matrices = []
# for file in files:
# 	df = utl.read_csv(path+file)
# 	data = utl.get_data_matrix(df)
# 	data_matrices.append(data)
# # print(data_matrices)
# cad = CoarseActivityDetection(data_matrices,450)
# cad.get_cmv_for_all_files()
# # cad.plot_cmvs()
# cad.get_max_variance()
# cad.plot_max_cmvs()


# cad.get_cmv_for_all_files()

##################

# path = '/Users/Apple/Documents/gitRepo/ra-eeyes-activity-detection/data_small/'
# utl = FileUtilities(path)
# amplitude_matrices = utl.get_amplitude_matrices()
# print(amplitude_matrices[0].shape)
# freq = np.zeros(40)
# for i in amplitude_matrices[0]:
# 	for j in i:
# 		freq[int(j)] += 1
# print(freq)
# plt.figure(figsize=(12,6))
# plt.title('Amplitude bins and counts')
# plt.xlabel('Bins')
# plt.ylabel('Amplitude Counts')
# plt.grid(True)
# pos = np.arange(freq.shape[0])
# width = 2.0
# ax = plt.axes()
# ax.set_xticks(pos)
# ax.set_xticklabels(pos)
# plt.bar(pos, freq, width)
# plt.show()

# print(amplitude_matrices[1].shape)
# freq = np.zeros(40)
# for i in amplitude_matrices[1]:
# 	for j in i:
# 		freq[int(j)] += 1
# print(freq)
# plt.figure(figsize=(12,6))
# plt.title('Amplitude bins and counts')
# plt.xlabel('Bins')
# plt.ylabel('Amplitude Counts')
# plt.grid(True)
# pos = np.arange(freq.shape[0])
# width = 1.0
# ax = plt.axes()
# ax.set_xticks(pos)
# ax.set_xticklabels(pos)
# plt.bar(pos, freq, width, color = 'g')
# plt.show()

##################

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
