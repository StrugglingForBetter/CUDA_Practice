#include <cuda_runtime.h>
#include <stdio.h>
#include <freshman.h>
#include <device_launch_parameters.h>

#define DIM 1024

int recursiveReduce(int* data, int const size) {
	if (size == 1)
		return data[0];
	int const stride = size / 2;
	if (size % 2 == 1) {
		for (int i = 0; i < stride; i++) {
			data[i] += data[i + stride];
		}
		data[0] += data[size - 1];
	}
	else {
		for (int i = 0; i < stride; i++) {
			data[i] += data[i + stride];
		}
	}

	return recursiveReduce(data, stride);
}

__global__ void warmup(int* g_idata, int* g_odata, unsigned int n) {
	// set thread ID
	unsigned int tid = threadIdx.x;

	// boundary check
	if (tid >= n)
		return;

	// convert global data pointer to the
	int* idata = g_idata + blockIdx.x * blockDim.x;
	// in-place reduction in global memory
	for (int stride = 1; stride < blockDim.x; stride *= 2) {
		if ((tid % (2 * stride)) == 0) {
			idata[tid] += idata[tid + stride];
		}

		// synchronize within block
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceGmem(int* g_idata, int* g_odata, unsigned int n) {
	// set thread ID
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

	// boundary check
	if (tid >= n)
		return;

	int* idata = g_idata + blockIdx.x * blockDim.x;

	// in-place reduction in global memory
	if (blockDim.x >= 1024 && tid < 512)
		idata[tid] += idata[tid + 512];
	__syncthreads();
	if (blockDim.x >= 512 && tid < 256)
		idata[tid] += idata[tid + 256];
	__syncthreads();
	if (blockDim.x >= 256 && tid < 128)
		idata[tid] += idata[tid + 128];
	__syncthreads();
	if (blockDim.x >= 128 && tid < 64)
		idata[tid] += idata[tid + 64];
	__syncthreads();

	// write result for this block to global mem
	if (tid < 32) {
		volatile int* vmem = idata;
		vmem[tid] += vmem[tid + 32];
		vmem[tid] += vmem[tid + 16];
		vmem[tid] += vmem[tid + 8];
		vmem[tid] += vmem[tid + 4];
		vmem[tid] += vmem[tid + 2];
		vmem[tid] += vmem[tid + 1];
	}

	if (tid == 0)
		g_odata[blockIdx.x] = idata[0];
}