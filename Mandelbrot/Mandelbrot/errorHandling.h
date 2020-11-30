#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstddef>
#include <stdexcept>

struct exception final : std::runtime_error
{
	explicit exception(cudaError_t const error) : std::runtime_error{ cudaGetErrorString(error) } {

	}
};

inline void check(cudaError_t const error) {
	if (error != cudaSuccess) {
		throw exception{ error };
	}
}
