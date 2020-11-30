#include "./kernel.cuh"
#include <iostream>

constexpr dim_t blocks(std::size_t const size, dim_t const tpb) noexcept {
	return static_cast <dim_t> ((size + tpb - 1) / tpb);
}

/* x = real part
*  x == 0 is at left edge
*  x goes left to right
*  y = img part
*  y == 0 is at lower edge
*  y goes lower to upper
*/

// get point for x and y
// lower left corner + span * percentage of completeion( between 0 and 1)
__device__ __host__ int value(int x, int y, const float leftReal, const float lowerImg, const float spanReal, const float spanImg) {
	float pointReal{ leftReal + spanReal * ((float)x / (float)width) };
	float pointImg{ lowerImg + spanImg * ((float)y / (float)height) };

	float zReal{ 0 };
	float zImg{ 0 };

	const int maxIterations{ 256 };
	int iterations{ 0 };

	while (((zReal * zReal + zImg * zImg) < 4) && iterations <= maxIterations) {
		float zReal2 = zReal * zReal, zImg2 = zImg * zImg;
		zImg = 2 * zReal * zImg + pointImg;
		zReal = zReal2 - zImg2 + pointReal;
		iterations++;
	}
	if (iterations < maxIterations) {
		return  iterations;
	}
	else return 0;
}

__global__ void mandel_kernel(float* leftReal, float* lowerImg, float* spanReal, float* spanImg, pfc::pixel_t puffer[]) {
	auto const t{ blockIdx.x * blockDim.x + threadIdx.x };
	if (t < height * width) {
		puffer[t] = { 0, pfc::byte_t(value(t % width, t / width, *leftReal, *lowerImg, *spanReal, *spanImg)),0 };
	}
}

cudaError_t call_mandel_kernel(float* leftReal, float* lowerImg, float* spanReal, float* spanImg, pfc::pixel_t puffer[], std::size_t size, dim_t tpb) {
	mandel_kernel << <blocks(size, tpb), tpb >> > (leftReal, lowerImg, spanReal, spanImg, puffer);
	return cudaGetLastError();
}
//threads pro block 32 increments 32 - 1024

