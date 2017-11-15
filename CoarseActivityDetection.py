#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
from Utilities import FileUtilities
from Utilities import MovingVariance
from matplotlib import pyplot as plt

class CoarseActivityDetection(object):
	"""
		Annotates and stores the the activity detected between walking and running
		Input: dataframes of the CSI data, threshold to be set.
		Output: Cumulatice Moving Variances
				Max of Cumulative Variances per data set
				plots Cumulatice Moving Variances
				plots Max of Cumulative Variances per data set
	"""
	
	def __init__(self, data_matrices, threshold):
		super(CoarseActivityDetection, self).__init__()
		self.data_matrices = data_matrices
		self.threshold = threshold
		self.max_cmvs_ = []
		self.cmvs_ = []
		print("CoarseActivityDetection initialized")

	def _store_cumulative_moving_variance(self, data):
		print('Calculating Cumulative Moving Variance...')
		amplitude = data[:,1:91]
		mv = MovingVariance(amplitude,1000)
		V = mv.get_moving_var()
		CMV = mv.get_cumulative_moving_variance(V)
		self.cmvs_.append(np.array(CMV))
		return self

	def get_cmv_for_all_files(self):
		print("Calculating cmv...")
		for data_matrix in self.data_matrices:
			self._store_cumulative_moving_variance(data_matrix)
		return self

	def get_max_variance(self):
		print(self.cmvs_)
		for cmv in self.cmvs_:
			self.max_cmvs_.append(np.nanmax(cmv,axis=0))
		return self

	def plot_cmvs(self):
		print("Plotting Cumulative Moving Variance...")
		plt.figure(figsize=(12,6))
		plt.title('Cumulative Moving Variance over time')
		plt.xlabel('Time')
		plt.ylabel('Cumulative Moving Variance')
		plt.grid(True)
		for cmv in self.cmvs_:
			plt.plot(range(1, cmv.shape[0]+1) , cmv)
		# plt.legend(loc='upper right')
		plt.show()

	def plot_max_cmvs(self):
		print("Plotting Maximum Cumulative Moving Variance...")
		plt.figure(figsize=(12,6))
		plt.title('Maximum Cumulative Moving Variance of each CSI set')
		plt.grid(True)
		plt.xlabel('Samples')
		plt.ylabel('Maximum Cumulative Moving Variance')
		print(self.max_cmvs_)
		# print(self.max_cmvs_.shape)
		plt.plot(range(1, len(self.max_cmvs_)+1) , self.max_cmvs_, marker = 'o')
		# plt.legend(loc='upper right')
		plt.show()