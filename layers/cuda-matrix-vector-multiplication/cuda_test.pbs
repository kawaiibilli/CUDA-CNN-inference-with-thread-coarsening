
NVCC = nvcc

matMult: gen_gpu.h gen_gpu.cu host.cu
	$(NVCC) -I ./ gen_gpu.cu host.cu -o matMult


run:
	./matMult

clean:
	rm -rf matMult

