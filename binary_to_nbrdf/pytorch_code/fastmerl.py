"""
:mod: `merl` -- MERL BRDF Python support
========================================

.. module:: merl
    :synopsis: This module implements the support for MERL BRDF material
               https://www.merl.com/brdf/

.. moduleauthor:: Alban Fichet <alban.fichet@inria.fr>

.. modified by Alejandro Sztrajman <a.sztrajman@ucl.ac.uk>
	- Implemented fast evaluation of multiple input angles in matrix form
	- Added function to write MERL binary files
	- Added multiple functions to interface with numpy arrays and pandas dataframes.
"""

import struct
import math
import numpy as np
import pandas as pd

class Merl:
	sampling_theta_h = 90
	sampling_theta_d = 90
	sampling_phi_d = 180

	scale = [1./1500, 1.15/1500, 1.66/1500]
    
	def __init__(self, merl_file):
		"""
		Initialize and load a MERL BRDF file

		:param merl_file: The path of the file to load
		"""
		with open(merl_file, 'rb') as f:
			data = f.read()
			n = struct.unpack_from('3i', data)
			self.sampling_phi_d = n[2]
			length = self.sampling_theta_h * self.sampling_theta_d * self.sampling_phi_d
			if  n[0]*n[1]*n[2] != length:
				raise IOError("Dimensions do not match")
			self.brdf = struct.unpack_from(str(3*length)+'d', data,	offset=struct.calcsize('3i'))
			self.brdf_np = np.array(self.brdf)

	def write_merl_file(self, fname):
		with open(fname, 'wb') as f:
			length = self.sampling_theta_h * self.sampling_theta_d * self.sampling_phi_d
			datastring = struct.pack('3i', self.sampling_theta_h, self.sampling_theta_d, self.sampling_phi_d)
			datastring += struct.pack(str(3*length)+'d', *self.brdf)
			f.write(datastring)

	def convert_to_fullmerl(self):
		if (self.sampling_phi_d == 180):
			brdf = np.array(self.brdf).reshape(3, self.sampling_theta_h, self.sampling_theta_d, self.sampling_phi_d)
			self.brdf = tuple(np.tile(brdf, 2).flatten())
			self.sampling_phi_d = 360
		else:
			print('sampling_phi_d != 180')

	def from_array(self, brdf_arr):
		if (brdf_arr.shape[-1] == 3):
			_arr = brdf_arr.T
		else:
			_arr = brdf_arr

		brdf_list = np.concatenate([_arr[0]/self.scale[0], _arr[1]/self.scale[1], _arr[2]/self.scale[2]])
		self.brdf = tuple(brdf_list)

	def from_dataframe(self, df, cols=['brdf_r', 'brdf_g', 'brdf_b']):
		self.from_array(df[cols].values.T)

	def to_dataframe(self, mode='interpolation'):
		cols = []
		for itheta_h in range(self.sampling_theta_h):
			theta_h = self.__theta_h_from_idx(itheta_h)
			for itheta_d in range(self.sampling_theta_d):
				theta_d = self.__theta_d_from_idx(itheta_d)
				for iphi_d in range(self.sampling_phi_d):
					phi_d = self.__phi_d_from_idx(iphi_d)
					cols += [[theta_h, theta_d, phi_d]]

		cols = np.array(cols)
		df = pd.DataFrame(cols, columns=['theta_h', 'theta_d', 'phi_d'])
		if (mode == 'raw'):
			_brdf = np.apply_along_axis(lambda row: self.eval_raw(*row), 1, cols)
		elif (mode == 'interpolation'):
			_brdf = np.apply_along_axis(lambda row: self.eval_interp(*row), 1, cols)

		df['brdf_r'] = _brdf.T[0]
		df['brdf_g'] = _brdf.T[1]
		df['brdf_b'] = _brdf.T[2]
		return df

	def eval_raw(self, theta_h, theta_d, phi_d):
		"""
		Lookup the BRDF value for given half diff coordinates
		
		:param theta_h: half vector elevation angle in radians
		:param theta_d: diff vector elevation angle in radians
		:param phi_d: diff vector azimuthal angle in radians
		:return: A list of 3 elements giving the BRDF value for R, G, B in
		linear RGB
		"""
		return self.__eval_idx(self.__theta_h_idx(theta_h),
							self.__theta_d_idx(theta_d),
							self.__phi_d_idx(phi_d))

	def __filter_phi_d(self, phi_d):
		if (self.sampling_phi_d == 180):
			phi_d = np.where(phi_d <= 0, phi_d + np.pi, phi_d)
		elif (self.sampling_phi_d == 360):
			phi_d = np.where(phi_d >= 2*np.pi, phi_d - 2*np.pi, phi_d)
			phi_d = np.where(phi_d < 0, phi_d + 2*np.pi, phi_d)
		return phi_d

	def eval_interp(self, theta_h, theta_d, phi_d):
		"""
		Lookup the BRDF value for given half diff coordinates and perform an
		interpolation over theta_h, theta_d and phi_d
		
		:param theta_h: half vector elevation angle in radians
		:param theta_d: diff vector elevation angle in radians
		:param phi_d: diff vector azimuthal angle in radians
		:return: A list of 3 elements giving the BRDF value for R, G, B in
		linear RGB
		"""
		phi_d = self.__filter_phi_d(phi_d)

		idx_th_p = self.__theta_h_idx(theta_h)
		idx_td_p = self.__theta_d_idx(theta_d)
		idx_pd_p = self.__phi_d_idx(phi_d)

		# Calculate the indexes for interpolation
		idx_th_p = np.where(idx_th_p < self.sampling_theta_h - 1, idx_th_p, self.sampling_theta_h - 2)
		idx_td_p = np.where(idx_td_p < self.sampling_theta_d - 1, idx_td_p, self.sampling_theta_d - 2)

		idx_th = [idx_th_p, idx_th_p + 1]
		idx_td = [idx_td_p, idx_td_p + 1]
		idx_pd = [idx_pd_p, idx_pd_p + 1]

		# Calculate the weights
		weight_th = [abs(self.__theta_h_from_idx(i) - theta_h) for i in idx_th]
		weight_td = [abs(self.__theta_d_from_idx(i) - theta_d) for i in idx_td]
		weight_pd = [abs(self.__phi_d_from_idx(i) - phi_d) for i in idx_pd]

		# Normalize the weights
		weight_th = [1 - w / sum(weight_th) for w in weight_th]
		weight_td = [1 - w / sum(weight_td) for w in weight_td]
		weight_pd = [1 - w / sum(weight_pd) for w in weight_pd]

		#idx_pd[1] = idx_pd[1] if idx_pd[1] < self.sampling_phi_d else 0
		idx_pd[1] = np.where(idx_pd[1] < self.sampling_phi_d, idx_pd[1], 0)

		ret_val = [0] * 3
		
		for ith, wth in zip(idx_th, weight_th):
			for itd, wtd in zip(idx_td, weight_td):
				for ipd, wpd in zip(idx_pd, weight_pd):
					ret_val = [r + x * wth * wtd * wpd
							for r, x, in zip(ret_val, self.__eval_idx(ith, itd, ipd))]

		return np.array(ret_val)

	def __eval_idx(self, ith, itd, ipd):
		"""
		Lookup the BRDF value for a given set of indexes
		:param ith: theta_h index
		:param itd: theta_d index
		:param ipd: phi_d index
		:return: A list of 3 elements giving the BRDF value for R, G, B in
		linear RGB
		"""
		ind = ipd + self.sampling_phi_d * (itd + ith * self.sampling_theta_d)

		stride = self.sampling_theta_h * self.sampling_theta_d * self.sampling_phi_d

		ret = []
		for color in range(0,3):
			ret += [self.brdf_np[(ind + color*stride).astype(np.int)] * self.scale[color]]

		return np.array(ret)
	
	def __theta_h_from_idx(self, theta_h_idx):
		"""
		Get the theta_h value corresponding to a given index

		:param theta_h_idx: Index for theta_h
		:return: A theta_h value in radians
		"""
		ret_val = theta_h_idx / self.sampling_theta_h
		return ret_val * ret_val * np.pi / 2

	def __theta_h_idx(self, theta_h):
		"""
		Get the index corresponding to a given theta_h value

		:param theta_h: Value for theta_h in radians
		:return: The corresponding index for the given theta_h
		"""
		th = self.sampling_theta_h * np.sqrt(theta_h / (np.pi/2))
		floorth = np.floor(th)
		return np.clip(floorth, 0, self.sampling_theta_h - 1)

	def __theta_d_from_idx(self, theta_d_idx):
		"""
		Get the theta_d value corresponding to a given index

		:param theta_d_idx: Index for theta_d
		:return: A theta_d value in radians
		"""
		return theta_d_idx / self.sampling_theta_d * np.pi / 2

	def __theta_d_idx(self, theta_d):
		"""
		Get the index corresponding to a given theta_d value

		:param theta_d: Value for theta_d in radians
		:return: The corresponding index for the given theta_d
		"""
		floortd = np.floor(self.sampling_theta_d * theta_d / (np.pi/2))
		return np.clip(floortd, 0, self.sampling_theta_d-1)

	def __phi_d_from_idx(self, phi_d_idx):
		"""
		Get the phi_d value corresponding to a given index

		:param phi_d_idx: Index for phi_d
		:return: A phi_d value in radians
		"""
		if (self.sampling_phi_d == 180):
			return phi_d_idx / self.sampling_phi_d * np.pi
		if (self.sampling_phi_d == 360):
			return phi_d_idx / self.sampling_phi_d * np.pi * 2

	def __phi_d_idx(self, phi_d):
		"""
		Get the index corresponding to a given phi_d value

		:param theta_h: Value for phi_d in radians
		:return: The corresponding index for the given phi_d
		"""
		phi_d = self.__filter_phi_d(phi_d)

		if (self.sampling_phi_d == 180):
			floorpd = np.floor(self.sampling_phi_d * phi_d / np.pi)
			return np.clip(floorpd, 0, self.sampling_phi_d - 1)
		if (self.sampling_phi_d == 360):
			floorpd = np.floor(self.sampling_phi_d * phi_d / (2*np.pi))
			return np.clip(floorpd, 0, self.sampling_phi_d - 1)

