NVCC = nvcc

conv_test: headers.h kernels.cu host.cu  
	$(NVCC) -I ./ kernels.cu host.cu -o conv2d

conv_run:
	./conv

clean:
	rm -rf conv