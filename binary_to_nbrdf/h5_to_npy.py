#!/usr/bin/env python3
import sys
import os
import os.path as op
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('h5s', nargs='+')
parser.add_argument('--destdir', default=None)
args = parser.parse_args()

import common
import keras
from keras import backend as K

def sequentialmodel_to_dir(mdl, savedir, matid):
	wdata = {
		'fc1':mdl.layers[0].get_weights()[0],
		'fc2':mdl.layers[1].get_weights()[0],
		'fc3':mdl.layers[2].get_weights()[0],
		'b1':mdl.layers[0].get_weights()[1],
		'b2':mdl.layers[1].get_weights()[1],
		'b3':mdl.layers[2].get_weights()[1],
	}
	os.popen('mkdir -p '+savedir).read()
	for key in wdata.keys():
		out_fname = op.join(savedir, matid+'_'+key+'.npy')
		print('Writing ', out_fname)
		np.save(out_fname, wdata[key])
    

for h5 in args.h5s:
	if args.destdir is None:
		destdir = op.join(op.dirname(h5), 'npy/')
	else:
		destdir = args.destdir

	matid = op.basename(h5).replace('.h5', '').replace('.json','')
	model = common.load_model(h5)
	sequentialmodel_to_dir(model, destdir, matid)
	K.clear_session()

