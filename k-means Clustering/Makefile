all: km_openmp km_pthreads

km_openmp: km_openmp.c
	gcc -O3 -march=native -o km_openmp -fopenmp km_openmp.c

km_pthreads: km_pthreads.c
	gcc -O3 -march=native -o km_pthreads -pthread km_pthreads.c 
