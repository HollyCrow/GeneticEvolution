run:
	/opt/cuda/bin/nvcc main.cu -lSDL2 -o "GE" && time ./GE