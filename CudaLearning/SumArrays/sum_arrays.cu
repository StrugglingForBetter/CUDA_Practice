#include <cuda_runtime.h>
#include <stdio.h>
#include <device_launch_parameters.h>
#include <malloc.h>
#include <string.h>

void sumArrays(float* a, float* b, float* res, const int size) {
	for (int i = 0; i < size; i += 4) {
		res[i] = a[i] + b[i];
		res[i + 1] = a[i + 1] + b[i + 1];
		res[i + 2] = a[i + 2] + b[i + 2];
		res[i + 3] = a[i + 3] + b[i + 3];
	}
}

__global__ void sumArraysGPU(float* a, float* b, float* res) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	res[i] = a[i] + b[i];
}

int main(int argc, char** argv) {

	int dev = 0;
	cudaSetDevice(dev);

	int nElem = 1 << 14;
	printf("Vector size: %d \n", nElem);
	int nByte = sizeof(float) * nElem;
	float* a_h = (float*)malloc(nByte);
	float* b_h = (float*)malloc(nByte);
	float* res_h = (float*)malloc(nByte);
	float* res_from_gpu_h = (float*)malloc(nByte);

	memset(res_h, 0, nByte);
	memset(res_from_gpu_h, 0, nByte);

	float* a_d, * b_d, * res_d;
	CHECK(cudaMalloc((float**)&a_d, nByte));

}