run:
	/opt/cuda/bin/nvcc main.cu -o "GE" -diag-suppress 177 -diag-suppress 549 -ccbin=/usr/bin/clang -lSDL2 && time ./GE


