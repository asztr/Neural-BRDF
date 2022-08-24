#!/usr/bin/env python3

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import os
from pathlib import Path

import tensorflow as tf
import keras
import keras.backend as K
import keras.optimizers
from keras.models import Sequential
from keras.layers import Dense, Lambda
from sklearn.utils import shuffle

import coords
import fastmerl
import common

Xvars = ['hx', 'hy', 'hz', 'dx', 'dy', 'dz']
Yvars = ['brdf_r', 'brdf_g', 'brdf_b']
loss_name = 'mean_absolute_logarithmic_error'
batch_size = 512
epochs = 100
verbose = 2
learning_rate = 5e-4

def mean_absolute_logarithmic_error(y_true, y_pred):
	return K.mean(K.abs(K.log(1 + y_true) - K.log(1 + y_pred)))

def brdf_to_rgb(rvectors, brdf):
	hx = K.reshape(rvectors[:, 0], (-1,1))
	hy = K.reshape(rvectors[:, 1], (-1,1))
	hz = K.reshape(rvectors[:, 2], (-1,1))
	dx = K.reshape(rvectors[:, 3], (-1,1))
	dy = K.reshape(rvectors[:, 4], (-1,1))
	dz = K.reshape(rvectors[:, 5], (-1,1))

	theta_h = tf.atan2(K.sqrt(hx**2 + hy**2), hz)
	theta_d = tf.atan2(K.sqrt(dx**2 + dy**2), dz)
	phi_d = tf.atan2(dy, dx)
	wiz = K.cos(theta_d)*K.cos(theta_h) - K.sin(theta_d)*K.cos(phi_d)*K.sin(theta_h)
	rgb = brdf*K.clip(wiz, 0, 1)
	return rgb

def loss_wrapper(input_tensor):
	def loss(y_true, y_pred):
		rgb_true = brdf_to_rgb(input_tensor, y_true)
		rgb_pred = brdf_to_rgb(input_tensor, y_pred)
		return mean_absolute_logarithmic_error(rgb_true, rgb_pred)
	return loss

def nn_model():
	model = Sequential()
	model.add(Dense(21, input_dim=6, kernel_initializer="random_uniform", activation='relu'))
	model.add(Dense(21, kernel_initializer="random_uniform", activation='relu'))
	model.add(Dense(3, kernel_initializer="random_uniform"))
	model.add(Lambda(lambda x_: K.exp(x_)-1))
	adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile(loss=loss_wrapper(model.input), optimizer=adam)
	return model

def train_nn(traindf, testdf, batch_size, epochs):
	trainX = traindf[Xvars].values
	trainY = traindf[Yvars].values

	testX = testdf[Xvars].values
	testY = testdf[Yvars].values

	model = nn_model()
	np.random.seed(0)
	history = model.fit(trainX, trainY, batch_size=batch_size, shuffle=True, epochs=epochs, verbose=verbose, validation_data=(testX, testY))
	score = model.evaluate(testX, testY, verbose=verbose)
	print('Score:', score)
	return model, history, score

def brdf_values(rvectors, brdf=None, model=None):
	if (brdf is not None):
		rangles = coords.rvectors_to_rangles(*rvectors)
		brdf_arr = brdf.eval_interp(*rangles).T
	elif (model is not None):
		brdf_arr = model.predict(rvectors.T)
	brdf_arr *= common.mask_from_array(rvectors.T).reshape(-1,1)
	return brdf_arr

def generate_nn_datasets(brdf, nsamples=800000, pct=0.8, seed=0):
	rangles = np.random.uniform([0,0,0],[np.pi/2.,np.pi/2.,2*np.pi],[int(nsamples*pct), 3]).T
	rangles[2] = common.normalize_phid(rangles[2])

	rvectors = coords.rangles_to_rvectors(*rangles)
	brdf_vals = brdf_values(rvectors, brdf=brdf)

	df = pd.DataFrame(np.concatenate([rvectors.T, brdf_vals], axis=1), columns=[*Xvars, *Yvars])
	df = df[(df.T != 0).any()]
	df = df.drop(df[df['brdf_r'] < 0].index)
	return df

def plot_loss_history(history, loss_name='', fname=None):
	val_loss = history.history['val_loss']
	loss = history.history['loss']
	n = len(loss)

	plt.plot(range(n), loss, color='r', label='loss')
	plt.plot(range(n), val_loss, color='b', label='val_loss')
	plt.xlabel('epoch')
	plt.ylabel(loss_name)
	plt.legend(loc='upper right')

	if (fname is not None):
		plt.savefig(fname, dpi=100, bbox_inches='tight')
		plt.clf()

def main(binary):
	basename = Path(binary).stem #output file are generated in the work folder

	#read data, generate datasets
	brdf = fastmerl.Merl(binary)
	traindf = generate_nn_datasets(brdf, nsamples=800000, pct=0.8, seed=2)
	testdf = generate_nn_datasets(brdf, nsamples=800000, pct=0.2, seed=3)

	#compute nn and plot loss
	lossplot_png = 'lossplot_'+basename+'.png'
	model, history, score = train_nn(traindf, testdf, batch_size=batch_size, epochs=epochs)
	plot_loss_history(history, loss_name=loss_name, fname=lossplot_png)
	print('wrote ', lossplot_png)

	#save model to file
	model_h5, model_json = basename+'.h5', basename+'.json'
	common.save_model(model, model_h5, model_json)
	print('wrote ', model_h5, model_json)

	K.clear_session()

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('binaries', nargs='+')
	parser.add_argument('--cuda_device', type=str, default='0')
	args = parser.parse_args()

	os.environ['CUDA_VISIBLE_DEVICES']=args.cuda_device
	common.keras_init_session(allow_growth=True)

	for binary in args.binaries:
		print(binary)
		main(binary)

