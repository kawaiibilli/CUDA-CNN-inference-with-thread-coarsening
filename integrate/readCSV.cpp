#include <vector>
#include <iostream>
#include <fstream>
#include <locale>
#include <limits>
#include <sstream>
#include <algorithm>
#include "headers.h"

using namespace std;

	FCLayer :: FCLayer (){

	}
	FCLayer :: FCLayer(std::string name, int inputs, int outputs) { 
		this->inputs = inputs;
		this->name = name; 
		this->outputs = outputs;
		weights = (float *)malloc((inputs*outputs) * sizeof(float )); 
		biases = (float *)malloc(outputs * sizeof(float));
	}

	ConvLayer :: ConvLayer(){

	}
	ConvLayer :: ConvLayer(std::string name, int filter_size, int num_layers, int depth) { 
		this->filter_size = filter_size;
		this->num_layers = num_layers;
		this->name = name;
		this->depth = depth;
		weights = (float *)malloc((filter_size*filter_size*num_layers*depth) * sizeof(float)); 
		biases = (float *)malloc(depth * sizeof(float));
	}


FCLayer processFC(std::stringstream& ss) {
	std::string name;
	int inputs;
	int outputs;
	getline( ss, name, ',' );
	std::string substr;
	getline( ss, substr, ',' );
	inputs = atoi(substr.c_str());
	getline( ss, substr, ',' );
	outputs = atoi(substr.c_str());
	FCLayer f(name, inputs, outputs);
	for (int i1 = 0 ; i1 < inputs; i1++) {
		for (int i2 = 0 ; i2 < outputs; i2++) {
			getline( ss, substr, ',' );
			float val = atof(substr.c_str());
			f.weights[i1*outputs + i2] = val;
		}
	}
	for (int i1 = 0 ; i1 < outputs; i1++) {
		getline( ss, substr, ',' );
		float val = atof(substr.c_str());
		f.biases[i1]= val;
	}
	return f;
}

ConvLayer processConv(std::stringstream& ss) {
	std::string name;
	int filter_size;
	int num_layers; 
	int depth;
	getline( ss, name, ',' );
	std::string substr;
	getline( ss, substr, ',' );
	filter_size = atoi(substr.c_str());
	getline( ss, substr, ',' );
	depth = atoi(substr.c_str());
	getline( ss, substr, ',' );
	num_layers = atoi(substr.c_str());
	ConvLayer c(name, filter_size, num_layers, depth);
	for (int i1 = 0 ; i1 < filter_size; i1++) {
		for (int i2 = 0 ; i2 < filter_size; i2++) {
			for (int i3 = 0 ; i3 < num_layers; i3++) {
				for (int i4 = 0 ; i4 < depth; i4++) {
					getline( ss, substr, ',' );
					float val = atof(substr.c_str());
					c.weights[i1*(filter_size*num_layers*depth) + i2*(num_layers*depth) + i3*depth + i4] = val;
				}
			}
		}
	}
	for (int i1 = 0 ; i1 < depth; i1++) {
		getline( ss, substr, ',' );
		float val = atof(substr.c_str());
		c.biases[i1]= val;
	}
	return c;
}
