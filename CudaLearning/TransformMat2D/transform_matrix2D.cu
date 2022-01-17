#include <cuda_runtime.h>
#include <stdio.h>
#include <freshman.h>
#include <device_launch_parameters.h>

void transformMatrix2D_CPU(float* MatA, float* MatB, int nx, int ny) {
	for (int j = 0; j < ny; j++) {
		for (int i = 0; i < nx; i++) {
			MatB[i * nx + j] = MatA[j * nx + i];
		}
	}
}

__global__ void copyRow(float* MatA, float* MatB, int nx, int ny) {

	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iy = threadIdx.y + blockDim.y * blockIdx.y;
	int idx = ix + iy * nx;
	if (ix < nx && iy < ny) {
		MatB[idx] = MatA[idx];
	}
}

__global__ void transformNaiveRow(float* MatA, float* MatB, int nx, int ny) {
	
	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iy = threadIdx.y + blockDim.y * blockIdx.y;

	int idx_row = ix + iy * nx;
	int idx_col = ix * ny + iy;
	if (ix < nx && iy < ny) {
		MatB[idx_col] = MatA[idx_row];
	}
}

__global__ void transformNaiveCol(float* MatA, float* MatB, int nx, int ny) {
	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iy = threadIdx.y + blockDim.y * blockIdx.y;

	int idx_row = ix + iy * nx;
	int idx_col = ix * ny + iy;

	if (ix < nx && iy < ny) {
		MatB[idx_row] = MatA[idx_col];
	}
}

__global__ void transformNaiveRowUnroll(float* MatA, float* MatB, int nx, int ny) {

	int ix = threadIdx.x + blockDim.x * blockIdx.x * 4;
	int iy = threadIdx.y + blockDim.y * blockIdx.y;
	int idx_row = ix + iy * nx;
	int idx_col = ix * ny + iy;

	if (ix < nx && iy < ny) {
		MatB[idx_col] = MatA[idx_row];

	}
}