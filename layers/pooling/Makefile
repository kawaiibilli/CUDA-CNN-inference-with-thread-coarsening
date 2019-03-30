NVCC = nvcc

helloCUDA: headers.h pooling.cu host.cu  
	$(NVCC) -I ./ pooling.cu host.cu -o helloCUDA

run: helloCUDA
	./helloCUDA

clean:
	rm -rf helloCUDA