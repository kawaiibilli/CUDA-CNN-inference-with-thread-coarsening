NVCC = nvcc

conv_test: headers.h conv.cu host.cu  
	$(NVCC) -I ./ conv.cu host.cu -o conv

conv_run:
	./conv

clean:
	rm -rf conv