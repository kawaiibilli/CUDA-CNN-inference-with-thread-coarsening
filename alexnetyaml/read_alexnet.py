import numpy as np

net = np.load("bvlc_alexnet.npy", encoding="bytes").item()
f = open("alexnet_noweight.yaml", "w")
for layer in net:
	f.write(layer + ":\n")
	if layer.find("fc") == -1: #Conv Layer
		filter_size = net[layer][0].shape[0]
		f.write("  filter_size: " + str(filter_size) + '\n')
		depth = net[layer][0].shape[3]
		f.write("  depth: " + str(depth) + '\n')
		num_layers = net[layer][0].shape[2]
		f.write("  num_layers: " + str(num_layers) + '\n')
		f.write("  weights: ")
		f.write("[")
		for i1 in range(0, filter_size):
			for i2 in range(0, filter_size):
				for i3 in range(0, num_layers):
					for i4 in range(0, depth):
						if (i1 == filter_size - 1 and i2 == filter_size - 1 and i3 == num_layers - 1 and i4 == depth - 1):
							f.write(str(net[layer][0][i1][i2][i3][i4]))
						else:
							f.write(str(net[layer][0][i1][i2][i3][i4]) + ", ")
		f.write("]")
		f.write('\n')
		f.write("  biases: ")
		f.write("[")
		for i1 in range(0, depth):
			if (i1 == depth - 1):
				f.write(str(net[layer][1][i1]))
			else: 
				f.write(str(net[layer][1][i1]) + ", ")
		f.write("]")	
		f.write('\n')
		
	else: #Fully Connected Layer
		inputs = net[layer][0].shape[0]
		f.write("  inputs: " + str(net[layer][0].shape[0]) + '\n')
		outputs = net[layer][1].shape[0]
		f.write("  outputs: " + str(net[layer][1].shape[0]) + '\n')
		f.write("  weights: ")
		f.write("[")
		for i1 in range(0, inputs):
			for i2 in range(0, outputs):
					if (i1 == inputs - 1 and i2 == outputs - 1 ):
						f.write(str(net[layer][0][i1][i2]))
					else:
						f.write(str(net[layer][0][i1][i2]) + ", ")
		f.write("]")
		f.write('\n')
		f.write("  biases: ")
		f.write("[")
		for i1 in range(0, outputs):
			if (i1 == outputs - 1):
				f.write(str(net[layer][1][i1]))
			else: 
				f.write(str(net[layer][1][i1]) + ", ")
		f.write("]")	
		f.write('\n')
	f.write('\n')
