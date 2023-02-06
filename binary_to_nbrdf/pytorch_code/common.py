from keras.models import model_from_json
import pyexr
import numpy as np
import coords

def save_model(model, h5, json=None):
	model.save_weights(h5)
	if (json == None):
		json = h5.replace('.h5', '.json')
	with open(json, 'w') as f:
		f.write(model.to_json())

def load_model(h5, json=None):
	if (json == None):
		json = h5.replace('.h5', '.json')
	with open(json, 'r') as f:
		model = model_from_json(f.read())
	model.load_weights(h5)
	return model

def normalize_phid(orig_phid):
	phid = orig_phid.copy()
	phid = np.where(phid < 0, phid + 2*np.pi, phid)
	phid = np.where(phid >= 2*np.pi, phid - 2*np.pi, phid)
	return phid

def mask_from_array(arr):
	if (len(arr.shape) > 1):
		mask = np.linalg.norm(arr, axis=1)
		mask[mask != 0] = 1
	else:
		mask = np.where(arr!=0, 1, 0)
	return mask

def keras_init_session(allow_growth=True, logging=False):
    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = allow_growth  # dynamically grow the memory used on the GPU
    config.log_device_placement = logging # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

