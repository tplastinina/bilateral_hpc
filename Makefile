NVCC = /usr/local/cuda/bin/nvcc
CFLAGS = -g -G -O0

EasyBMP.o: ./bmp/EasyBMP.cpp
	g++ -O3 -c $< -o $@

LNK = ./bmp/EasyBMP.h ./bmp/EasyBMP.cpp
biliteral: kernel.cu 
	$(NVCC) $(CFLAGS) $< -o $@ EasyBMP.o

clear:
	rm output.bmp