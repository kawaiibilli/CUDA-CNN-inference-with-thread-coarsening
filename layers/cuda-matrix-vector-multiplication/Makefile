
NVCC = nvcc

helloCUDA: gen_gpu.h gen_gpu.cu host.cu
	$(NVCC) -I ./ gen_gpu.cu host.cu -o helloCUDA


run:
	./helloCUDA

clean:
	rm -rf helloCUDA
