#include "./kernel.cuh"
#include <iostream>

/* x = real part
*  x == 0 is at left edge
*  x goes left to right
*  y = img part
*  y == 0 is at lower edge
*  y goes lower to upper
*/

// get point for x and y
// lower left corner + span * percentage of completeion( between 0 and 1)
__device__ int value(const int x, const int y, const float* leftReal, const float* lowerImg, const float* spanReal, const float* spanImg) {
	const float pointReal{ *leftReal + *spanReal * ((float)x / (float)width) };
	const float pointImg{ *lowerImg + *spanImg * ((float)y / (float)height) };

	float zReal{ 0 };
	float zImg{ 0 };
	float zReal2{ 0 };
	float zImg2{ 0 };

	const int maxIterations{ 256 };
	int iterations{ 0 };
	while (((zReal2 + zImg2) < 4) && iterations <= maxIterations) {
		zImg = 2 * zReal * zImg + pointImg;
		zReal = zReal2 - zImg2 + pointReal;
		iterations++;
		zReal2 = zReal * zReal, zImg2 = zImg * zImg;
	}
	return iterations;
}

__global__ void
mandel_kernel(float* leftReal, float* lowerImg, float* spanReal, float* spanImg, pfc::pixel_t puffer[]) {
	auto const x{ blockIdx.x * blockDim.x + threadIdx.x };
	auto const y{ blockIdx.y * blockDim.y + threadIdx.y };
	if (x * y < height * width) {
		puffer[x + y * width] = { 0,
			pfc::byte_t(value(x, y, leftReal, lowerImg, spanReal, spanImg)),
			0 };
	}
}

cudaError_t call_mandel_kernel(float* leftReal, float* lowerImg, float* spanReal, float* spanImg, pfc::pixel_t puffer[], std::size_t size) {
	const dim3 threadsPerBlock(16, 16);
	const dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);
	mandel_kernel << <numBlocks, threadsPerBlock >> > (leftReal, lowerImg, spanReal, spanImg, puffer);
	return cudaGetLastError();
}
//threads pro block 32 increments 32 - 1024

