#include <cuda_runtime.h>
#include <stdio.h>
#include <freshman.h>
#include <device_launch_parameters.h>
#include <math.h>
#define N 3000000

__global__ void kernel_1() {
	double sum = 0.0;
	for (int i = 0; i < N; i++)
		sum = sum + tan(0.1) * tan(0.1);
}

__global__ void kernel_2() {
	double sum = 0.0;
	for (int i = 0; i < N; i++)
		sum = sum + tan(0.1) * tan(0.1);
}

__global__ void kernel_3() {
	double sum = 0.0;
	for (int i = 0; i < N; i++)
		sum = sum + tan(0.1) * tan(0.1);
}

__global__ void kernel_4() {
	double sum = 0.0;
	for (int i = 0; i < N; i++)
		sum = sum + tan(0.1) * tan(0.1);
}

int main() {

	/*setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32", 1);*/  // windows����û��setenv, ��������Linuxϵͳ
	putenv("CUDA_DEVICE_MAX_CONNECTIONS");

	int dev = 0;
	cudaSetDevice(dev);
	int n_stream = 16;
	cudaStream_t* stream = (cudaStream_t*)malloc(n_stream * sizeof(cudaStream_t));

	for (int i = 0; i < n_stream; i++) {
		cudaStreamCreate(&stream[i]);
	}

	dim3 block(1);
	dim3 grid(1);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	for (int i = 0; i < n_stream; i++) {
		kernel_1 << <grid, block, 0, stream[i] >> > ();
		kernel_2 << <grid, block, 0, stream[i] >> > ();
		kernel_3 << <grid, block, 0, stream[i] >> > ();
		kernel_4 << <grid, block, 0, stream[i] >> > ();
	}

	cudaEventRecord(stop, 0);
	CHECK(cudaEventSynchronize(stop));

	float elapsed_time;
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("elapsed time: %f ms \n", elapsed_time);

	for (int i = 0; i < n_stream; i++) {
		cudaStreamDestroy(stream[i]);
	}

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	free(stream);
	CHECK(cudaDeviceReset());
	return 0;
}