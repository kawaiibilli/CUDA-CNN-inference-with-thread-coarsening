#include <vector>
#include <iostream>
#include <fstream>
#include <locale>
#include <limits>
#include <sstream>
#include <algorithm>
#include "headers.h"

class FCLayer {
public:
	std::string name;
	int inputs;
	int outputs;
	float* weights;	// dim - inputs x outputs 
	float* biases;
	FCLayer(std::string name, int inputs, int outputs) { 
		weights = (float *)malloc((inputs*outputs) * sizeof(float )); 
		biases = (float *)malloc(outputs * sizeof(float));
	}
};

class ConvLayer {
public:
	std::string name;
	int filter_size;
	int num_layers; 
	int depth;
	float* weights;	// dim - filter_size x filter_size x num_layers x depth
	float* biases;
	ConvLayer(std::string name, int filter_size, int num_layers, int depth) { 
		weights = (float *)malloc((filter_size*filter_size*num_layers*depth) * sizeof(float)); 
		biases = (float *)malloc(depth * sizeof(float));
	}
};

FCLayer processFC(std::stringstream& ss) {
	std::string name;
	int inputs;
	int outputs;
	getline( ss, name, ',' );
	std::string substr;
	getline( ss, substr, ',' );
	inputs = stoi(substr);
	getline( ss, substr, ',' );
	outputs = stoi(substr);
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
	filter_size = stoi(substr);
	getline( ss, substr, ',' );
	depth = stoi(substr);
	getline( ss, substr, ',' );
	num_layers = stoi(substr);
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

int main() {
	std::string line;
	std::ifstream in("../alexnet.csv");
	while(getline(in, line)) {
		std::stringstream lineStream(line);
    	if (line[0] == 'c') { //Convolutional layer 
    		ConvLayer c = processConv(lineStream);
    	} else if (line[0] == 'f') { //Fully Connected Layer
    		FCLayer f = processFC(lineStream);
    	}
    }
}
