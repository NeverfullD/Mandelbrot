#pragma once
#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#include "./bitmap.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstddef>
#include <cfloat>

using dim_t = decltype(dim3::x);

const int width = 8192;
const int height = 4608;

cudaError_t call_mandel_kernel(float* leftReal, float* lowerImg, float* spanReal, float* spanImg, pfc::pixel_t puffer[], std::size_t size);
