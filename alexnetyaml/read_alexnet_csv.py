import numpy as np



net = np.load("bvlc_alexnet.npy", encoding="bytes").item()
f = open("alexnet.csv", "w")
for layer in net:
	num_iters = 0
	f.write(layer + ", ")
	if layer.find("fc") == -1: #Conv Layer
		filter_size = net[layer][0].shape[0]
		f.write(str(filter_size) + ", ");
		depth = net[layer][0].shape[3]
		f.write(str(depth) + ", ")
		num_layers = net[layer][0].shape[2]
		f.write(str(num_layers) + ", ")
		for i1 in range(0, filter_size):
			for i2 in range(0, filter_size):
				for i3 in range(0, num_layers):
					for i4 in range(0, depth):
							f.write(str(net[layer][0][i1][i2][i3][i4]) + ", ")
							num_iters = num_iters + 1
		for i1 in range(0, depth):
			if (i1 == depth - 1):
				f.write(str(net[layer][1][i1]))
				num_iters = num_iters + 1
			else: 
				f.write(str(net[layer][1][i1]) + ", ")
				num_iters = num_iters + 1
		print layer, filter_size, filter_size, num_layers, depth
		print layer, num_iters, filter_size * filter_size * num_layers * depth + depth
	else: #Fully Connected Layer
		inputs = net[layer][0].shape[0]
		f.write(str(inputs) + ", ");
		outputs = net[layer][1].shape[0]
		f.write(str(outputs) + ", ");
		for i1 in range(0, inputs):
			for i2 in range(0, outputs):
				f.write(str(net[layer][0][i1][i2]) + ", ")
				num_iters = num_iters + 1
		for i1 in range(0, outputs):
			if (i1 == outputs - 1):
				f.write(str(net[layer][1][i1]))
				num_iters = num_iters + 1
			else: 
				f.write(str(net[layer][1][i1]) + ", ")
				num_iters = num_iters + 1
		print layer, inputs, outputs
		print layer, num_iters
	f.write('\n')