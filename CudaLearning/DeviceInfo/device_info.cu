#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char** argv) {

	printf("%s Starting ...\n", argv[0]);
	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
	if (error_id != cudaSuccess) {
		printf("cudaGetDeviceCount returned %d \n -> %s \n",
			(int)error_id, cudaGetErrorString(error_id));
		printf("Result = FAIL \n");
		exit(EXIT_FAILURE);
	}

	if (deviceCount == 0) {
		printf("There are no available devices that support CUDA \n");
	}
	else {
		printf("Detected %d CUDA Capable device(s) \n", deviceCount);
	}

	int dev = 0, driverVersion = 0, runtimeVersion = 0;
	cudaSetDevice(dev);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	printf("Device %d:\" %s\"\n", dev, deviceProp.name);
	cudaDriverGetVersion(&driverVersion);
	cudaRuntimeGetVersion(&runtimeVersion);
	printf("	CUDA Driver Version / Runtime Version	%d.%d / %d.%d\n",
		driverVersion / 1000, (driverVersion % 100) / 10, runtimeVersion / 1000, (runtimeVersion % 100) / 10);
	printf("	Total amount of global memory:			%.2f GBytes (%llu bytes)\n",
		(float)deviceProp.totalGlobalMem / pow(1024.0, 3), deviceProp.totalGlobalMem);
	printf("	GPU Clock rate:							%.0f MHZ (%0.2f GHZ) \n",
		deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
	printf("	Memory Bus width:						%d-bits\n", deviceProp.memoryBusWidth);
	if (deviceProp.l2CacheSize) {
		printf("	L2 Cache Size:						%d bytes\n", deviceProp.l2CacheSize);
	}
	printf("	Max Texture Dimension Size (x,y,z)            1D=(%d),2D=(%d,%d),3D=(%d,%d,%d)\n",
		deviceProp.maxTexture1D, deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1]
		, deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);

	printf("	Max Layered Texture Size (dim) x layers			1D=(%d) x %d, 2D=(%d,%d) x %d\n",
		deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1],
		deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
		deviceProp.maxTexture2DLayered[2]);
	printf("	Total amount of constant memory					%lu bytes\n", deviceProp.sharedMemPerBlock);
	printf("	Total number of registers available per block: %d \n", deviceProp.regsPerBlock);
	printf("	Wrap Size:										%d \n", deviceProp.warpSize);
	printf("	Maximum number of thread per multiprocesser:	%d \n", deviceProp.maxThreadsPerMultiProcessor);
	printf("	Maximum size of each dimension of a block:		%d x %d x %d \n",
		deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
	printf("	Maximum size of each dimension of a grid:		%d x %d x %d \n",
		deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
	printf("	Maximum memory pitch							%lu bytes \n", deviceProp.memPitch);

}