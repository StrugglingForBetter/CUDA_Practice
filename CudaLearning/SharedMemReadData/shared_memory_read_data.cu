#include <cuda_runtime.h>
#include <stdio.h>
#include <freshman.h>
#include <device_launch_parameters.h>

#define BDIMX 32
#define BDIMY 32

#define BDIMX_RECT 32
#define BDIMY_RECT 16
#define IPAD 1


__global__ void warmup(int* out) {
	__shared__ int tile[BDIMY][BDIMX];
	unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

	tile[threadIdx.y][threadIdx.x] = idx;
	__syncthreads();
	out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setRowReadRow(int* out) {
	__shared__ int tile[BDIMY][BDIMX];
	unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

	tile[threadIdx.y][threadIdx.x] = idx;
	__syncthreads();
	out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setColReadCol(int* out) {
	__shared__ int tile[BDIMY][BDIMX];
	unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

	tile[threadIdx.x][threadIdx.y] = idx;
	__syncthreads();
	out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setColReadRow(int* out) {
	__shared__ int tile[BDIMY][BDIMX];
	unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

	tile[threadIdx.x][threadIdx.y] = idx;
	__syncthreads();
	out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setRowReadCol(int* out) {
	__shared__ int tile[BDIMY][BDIMX];
	unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

	tile[threadIdx.y][threadIdx.x] = idx;
	__syncthreads();
	out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setRowReadColDyn(int* out) {
	extern __shared__ int tile[];
	unsigned int row_idx = threadIdx.y * blockDim.x + threadIdx.x;
	unsigned int col_idx = threadIdx.x * blockDim.y + threadIdx.y;
	tile[row_idx] = row_idx;
	__syncthreads();
	out[row_idx] = tile[col_idx];
}

__global__ void setRowReadColIpad(int* out) {
	__shared__ int tile[BDIMY][BDIMX + IPAD];
	
}