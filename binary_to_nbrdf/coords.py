import numpy as np

def rotate_vector(v, axis, angle):
	return v*np.cos(angle) + axis*dot(axis, v)*(1 - np.cos(angle)) + cross(axis, v)*np.sin(angle)

def io_to_hd(wi, wo):
	#compute halfway vector
	half = normalize(*(wi + wo))
	r_h, theta_h, phi_h = xyz2sph(*half)

	#compute diff vector
	bi_normal = np.tile([0.0, 1.0, 0.0], (wi.shape[1], 1)).T
	normal = np.tile([0.0, 0.0, 1.0], (wi.shape[1], 1)).T
	tmp = rotate_vector(wi, normal, -phi_h)
	diff = rotate_vector(tmp, bi_normal, -theta_h)
	return half, diff

def hd_to_io(half, diff):
	r_h, theta_h, phi_h = xyz2sph(*half)

	y_axis = np.tile([0.0, 1.0, 0.0], (half.shape[1], 1)).T
	z_axis = np.tile([0.0, 0.0, 1.0], (half.shape[1], 1)).T

	tmp = rotate_vector(diff, y_axis, theta_h)
	wi = normalize(*rotate_vector(tmp, z_axis, phi_h))
	wo = normalize(*(2*dot(wi, half)*half - wi))
	return wi, wo

def dot(v1, v2):
	return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]

def cross(v1, v2):
	return np.cross(v1.T, v2.T).T

def xyz2sph(x, y, z):
	r2_xy = x**2 + y**2
	r = np.sqrt(r2_xy + z**2)
	theta = np.arctan2(np.sqrt(r2_xy), z)
	phi = np.arctan2(y, x)
	return np.array([r, theta, phi])

def normalize(x, y, z):
	norm = np.sqrt(x**2 + y**2 + z**2)
	norm = np.where(norm == 0, np.inf, norm)
	return np.array([x, y, z]) / norm

def sph2xyz(r, theta, phi):
	x = r*np.sin(theta)*np.cos(phi)
	y = r*np.sin(theta)*np.sin(phi)
	z = r*np.cos(theta)
	return np.array([x, y, z])

#assumes phi_h=0 and both norms=1
def rangles_to_rvectors(theta_h, theta_d, phi_d):
	hx = np.sin(theta_h)*np.cos(0.0)
	hy = np.sin(theta_h)*np.sin(0.0)
	hz = np.cos(theta_h)
	dx = np.sin(theta_d)*np.cos(phi_d)
	dy = np.sin(theta_d)*np.sin(phi_d)
	dz = np.cos(theta_d)
	return np.array([hx, hy, hz, dx, dy, dz])

def rsph_to_rvectors(half_sph, diff_sph):
	hx, hy, hz = sph2xyz(*half_sph)
	dx, dy, dz = sph2xyz(*diff_sph)
	return np.array([hx, hy, hz, dx, dy, dz])

def rvectors_to_rsph(hx, hy, hz, dx, dy, dz):
	half_sph = xyz2sph(hx, hy, hz)
	diff_sph = xyz2sph(dx, dy, dz)
	return half_sph, diff_sph

def rvectors_to_rangles(hx, hy, hz, dx, dy, dz):
	theta_h = np.arctan2(np.sqrt(hx**2 + hy**2), hz)
	theta_d = np.arctan2(np.sqrt(dx**2 + dy**2), dz)
	phi_d = np.arctan2(dy, dx)
	return np.array([theta_h, theta_d, phi_d])
