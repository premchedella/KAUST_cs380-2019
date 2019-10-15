#include "imageprocessing.cuh"
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <math.h>
#include <sstream>

// TODO: read about the CUDA programming model: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model
// If everything is setup correctly, this file is compiled by the CUDA/C++ compiler (that is different from the C++ compiler).
// The CUDA/C++ compiler understands certain things that your C++ compiler doesn't understand - like '__global__', 'threadIdx', and function calls with triple-angle brackets, e.g., testArray<<<...>>>();


// do not use this method for anything else than verifying cuda compiled, linked and executed
__global__ void testArray(float* dst, float value) {
	unsigned int index = threadIdx.x;
	dst[index] = value;
}

void testCudaCall() {
	// quick and dirty test of CUDA setup
	const unsigned int N = 1024;
	float* device_array;
	cudaMalloc(&device_array, N * sizeof(float));
	testArray << <1, N >> > (device_array, -0.5f);
	float x[N];
	cudaMemcpy(x, device_array, N * sizeof(float), cudaMemcpyDeviceToHost);
	std::cout << "quick and dirty test of CUDA setup: " << x[0] << " " << x[1] << " " << x[1023] << std::endl;
	cudaFree(device_array);
}



// TODO: implement the image processing operations using CUDA kernels
