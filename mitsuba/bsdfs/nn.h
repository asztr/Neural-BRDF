#ifndef NN_H
#define NN_H

#include <iostream>
#include <memory>
#include <vector>
#include <random>
#include <string>
#include "npy.hpp"

using namespace std;

typedef vector<vector<float>> vector2D;
typedef vector<float> vector1D;

vector2D to_vector2D(vector1D vec, vector<unsigned long> shape) {
	vector2D ret(shape[0], vector1D(shape[1], 0));
	for(unsigned int i=0;i<shape[0];i++)
		for(unsigned int j=0;j<shape[1];j++)
			ret[i][j] = vec[i*shape[1] + j];
	return ret;
}

vector2D read_2Dnpy(string fname) {
	vector<unsigned long> shape;
	vector1D arr;
	npy::LoadArrayFromNumpy(fname, shape, arr);
	return to_vector2D(arr, shape);
}

vector1D read_1Dnpy(string fname) {
	vector<unsigned long> shape;
	vector1D arr;
	npy::LoadArrayFromNumpy(fname, shape, arr);
	return arr;
}

class Net {
	private:
		unsigned int nlayers = 4;
		unsigned int layers[4] = {6, 21, 21, 3};
		vector2D fc1, fc2, fc3;
		vector1D b1, b2, b3;

	public:
		Net();
		Net(string basename);
		vector1D forward(vector1D input) const;
};

Net::Net() {
}

Net::Net(string basename) {
	fc1 = read_2Dnpy(basename+"fc1.npy");
	fc2 = read_2Dnpy(basename+"fc2.npy");
	fc3 = read_2Dnpy(basename+"fc3.npy");
	b1 = read_1Dnpy(basename+"b1.npy");
	b2 = read_1Dnpy(basename+"b2.npy");
	b3 = read_1Dnpy(basename+"b3.npy");
}

vector1D Net::forward(vector1D input) const {
	float a1[layers[1]] = {};
	float a2[layers[2]] = {};
	vector1D a3(layers[3], 0.0);

	for(unsigned int i=0;i<layers[0];i++)
		for(unsigned int j=0;j<layers[1];j++)
			a1[j] += input[i]*fc1[i][j];

	for(unsigned int i=0;i<layers[1];i++)
		a1[i] = std::max(a1[i]+b1[i], (float)0.0);

	for(unsigned int i=0;i<layers[1];i++)
		for(unsigned int j=0;j<layers[2];j++)
			a2[j] += a1[i]*fc2[i][j];

	for(unsigned int i=0;i<layers[2];i++)
		a2[i] = std::max(a2[i]+b2[i], (float)0.0);

	for(unsigned int i=0;i<layers[2];i++)
		for(unsigned int j=0;j<layers[3];j++)
			a3[j] += a2[i]*fc3[i][j];

	for(unsigned int i=0;i<layers[3];i++)
		a3[i] = std::max(exp(a3[i] + b3[i]) - 1.0, 0.0);

	return a3;
}

class interp1D {
	private:
		vector1D x;
		vector1D y;
		float dx;
	public:
		interp1D();
		interp1D(vector1D _x, vector1D _y);
		interp1D(vector1D _y, float x0, float xf);
		float eval_raw(float x);
		float eval_interp(float x);
};

interp1D::interp1D() {
}

interp1D::interp1D(vector1D _x, vector1D _y) {
	dx = (_x.back() - _x[0]) / (_y.size() - 1);
	x = _x;
	y = _y;
}

interp1D::interp1D(vector1D _y, float x0, float xf) {
	dx = (xf - x0) / (_y.size() - 1);
	for(float _x=0.0;_x<=xf; _x += dx)
		x.push_back(_x);
	y = _y;
}

float interp1D::eval_raw(float _x) {
	unsigned int i = int((_x - x[0]) / dx);
	return y[i];
}

float interp1D::eval_interp(float _x) {
	unsigned int i = int((_x - x[0]) / dx);
	if (i == x.size()-1)
		return y.back();
	else {
		float t = (_x - x[i]) / dx;
		return y[i]*(1.0-t) + y[i+1]*t;
	}
}

#endif
