NVCC = nvcc

cnn_inference: headers.h pooling.cu conv.cu gen_gpu.cu host_main.cu  readCSV.cpp
	$(NVCC) -I ./ readCSV.cpp pooling.cu conv.cu gen_gpu.cu host_main.cu -o cnn_inference

run: cnn_inference
	./cnn_inference

clean:
	rm -rf cnn_inference