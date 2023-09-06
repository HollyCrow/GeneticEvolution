run:
	/opt/cuda/bin/nvcc main.cu -lSDL2 -o "GE" -diag-suppress 177 && time ./GE