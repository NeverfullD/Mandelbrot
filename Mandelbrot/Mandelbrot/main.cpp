

#include "./bitmap.h"
#include "./jobs.h"
#include "./kernel.cuh"
#include "./errorHandling.h"

#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <future>

using namespace std::chrono_literals;
//Control structures
const auto save_images{ false };
auto const cpu_seq{ false };
auto const cpu_thread{ false };
auto const cpu_task{ false };
auto const gpu{ true };

const auto jobFile{ "./jobs-200.txt" };
jobs <> const j{ jobFile };
const auto size = j.size();

template <typename... T>
double to_seconds(std::chrono::duration<T...> const duration)
{
	return std::chrono::duration<double, std::ratio<1>>{duration}.count();
}

template <typename... A, std::invocable<A...> F>
auto timed_run(F&& f, A &&... a)
{
	using clock_type = std::chrono::high_resolution_clock;

	auto const start = clock_type::now();
	std::invoke(std::forward<F>(f), std::forward<A>(a)...);
	auto const stop = clock_type::now();
	return stop - start;
}

// Mandelbrot
void mandelOnePicture(const int index, const float leftReal, const float lowerImg, const float spanReal, const float spanImg) {
	pfc::bitmap bmp{ width, height };
	auto& span{ bmp.pixel_span() };
	auto* const p_buffer{ std::data(span) };

	for (int y{ 0 }; y < height; ++y) {
		for (int x{ 0 }; x < width; ++x) {
			p_buffer[y * width + x] = {
			  0, pfc::byte_t(value(x, y, leftReal, lowerImg, spanReal, spanImg)),  0
			};
		}
	}

	//save mandel image
	if (save_images) {
		bmp.to_file("./mandel-" + std::to_string(index) + ".bmp");
	}
}

void calc_on_cpu_seq() {
	std::cout << "start cpu seq" << std::endl;
	float leftReal{  };
	float lowerImg{  };

	float spanReal{  };
	float spanImg{  };

	for (int i = 0; i < size; i++)
	{
		leftReal = real(j.get_lower_left(i));
		lowerImg = imag(j.get_lower_left(i));

		spanReal = real(j.get_size(i));
		spanImg = imag(j.get_size(i));

		mandelOnePicture(i, leftReal, lowerImg, spanReal, spanImg);
	}
}

void calc_on_cpu_threads() {
	std::cout << "start cpu threads" << std::endl;
	std::vector<std::thread> threads = {};
	threads.resize(size);

	float leftReal{  };
	float lowerImg{  };

	float spanReal{  };
	float spanImg{  };

	for (int i = 0; i < size; i++)
	{
		leftReal = real(j.get_lower_left(i));
		lowerImg = imag(j.get_lower_left(i));

		spanReal = real(j.get_size(i));
		spanImg = imag(j.get_size(i));
		threads[i] = std::thread{ mandelOnePicture, i, leftReal, lowerImg, spanReal, spanImg };
	}

	for (int i = size - 1; i >= 0; i--)
	{
		threads[i].join();
	}
}

void calc_on_cpu_tasks() {
	std::cout << "start cpu task" << std::endl;
	std::vector<std::future<void>> tasks = {};
	tasks.resize(size);

	float leftReal{  };
	float lowerImg{  };

	float spanReal{  };
	float spanImg{  };

	for (int i = 0; i < size; i++)
	{
		leftReal = real(j.get_lower_left(i));
		lowerImg = imag(j.get_lower_left(i));

		spanReal = real(j.get_size(i));
		spanImg = imag(j.get_size(i));
		tasks[i] = std::async(std::launch::async, mandelOnePicture, i, leftReal, lowerImg, spanReal, spanImg);
	}

	for (int i = size - 1; i >= 0; i--)
	{
		tasks[i].get();
	}
}

void calc_on_gpu() {
	std::cout << "start gpu" << std::endl;

	int count{ -1 };
	check(cudaGetDeviceCount(&count));

	if (count > 0)
	{
		check(cudaSetDevice(0));
		pfc::bitmap bmp{ width, height };
		auto& span{ bmp.pixel_span() };
		auto* p_buffer{ std::data(span) };

		float hLeftReal{  };
		float hLowerImg{  };

		float hSpanReal{  };
		float hSpanImg{  };

		float* dLeftReal{};
		check(cudaMalloc(&dLeftReal, sizeof(float)));
		float* dLowerImg{};
		check(cudaMalloc(&dLowerImg, sizeof(float)));
		float* dSpanReal{};
		check(cudaMalloc(&dSpanReal, sizeof(float)));
		float* dSpanImg{};
		check(cudaMalloc(&dSpanImg, sizeof(float)));

		pfc::pixel_t* dPuffer{};
		check(cudaMalloc(&dPuffer, width * height * sizeof(pfc::pixel_t)));

		for (int i = 0; i < size; i++)
		{
			hLeftReal = real(j.get_lower_left(i));
			hLowerImg = imag(j.get_lower_left(i));

			hSpanReal = real(j.get_size(i));
			hSpanImg = imag(j.get_size(i));

			check(cudaMemcpy(dLeftReal, &hLeftReal, sizeof(float), cudaMemcpyHostToDevice));
			check(cudaMemcpy(dLowerImg, &hLowerImg, sizeof(float), cudaMemcpyHostToDevice));
			check(cudaMemcpy(dSpanReal, &hSpanReal, sizeof(float), cudaMemcpyHostToDevice));
			check(cudaMemcpy(dSpanImg, &hSpanImg, sizeof(float), cudaMemcpyHostToDevice));

			call_mandel_kernel(dLeftReal, dLowerImg, dSpanReal, dSpanImg, dPuffer, width * height, 128);

			check(cudaMemcpy(p_buffer, dPuffer, width * height * sizeof(pfc::pixel_t), cudaMemcpyDeviceToHost));

			//save mandel image
			if (save_images) {
				bmp.to_file("./mandel-" + std::to_string(i) + ".bmp");
			}
		}
		check(cudaFree(dPuffer));
		check(cudaFree(dSpanImg));
		check(cudaFree(dSpanReal));
		check(cudaFree(dLowerImg));
		check(cudaFree(dLeftReal));
	}
	check(cudaDeviceReset());
}

int main() {
	std::chrono::nanoseconds elapsed_cpu_seq;
	std::chrono::nanoseconds elapsed_cpu_thread;
	std::chrono::nanoseconds elapsed_cpu_task;
	std::chrono::nanoseconds elapsed_gpu;

	//CPU_seq
	if (cpu_seq)
		elapsed_cpu_seq = timed_run([] {calc_on_cpu_seq(); });

	//CPU_thread
	if (cpu_thread)
		elapsed_cpu_thread = timed_run([] {calc_on_cpu_threads(); });

	//CPU_task
	if (cpu_task)
		elapsed_cpu_task = timed_run([] {calc_on_cpu_tasks(); });

	//GPU		   
	if (gpu)
		elapsed_gpu = timed_run([] {calc_on_gpu(); });

	std::cout << "number of pictures: " << j.size() << std::endl;
	std::cout << "====== times =====" << std::endl;
	if (cpu_seq)
		std::cout << "cpu seq time: " << to_seconds(elapsed_cpu_seq) << " s" << std::endl;

	if (cpu_thread)
		std::cout << "cpu thread time: " << to_seconds(elapsed_cpu_thread) << " s" << std::endl;

	if (cpu_task)
		std::cout << "cpu task time: " << to_seconds(elapsed_cpu_task) << " s" << std::endl;

	if (gpu)
		std::cout << "gpu time: " << to_seconds(elapsed_gpu) << " s" << std::endl;

	std::cout << "====== Speed Up =====" << std::endl;
	if (cpu_seq && cpu_thread)
		std::cout << "speedup cpu seq to cpu thread: " << to_seconds(elapsed_cpu_seq) / to_seconds(elapsed_cpu_thread) << std::endl;

	if (cpu_seq && cpu_task)
		std::cout << "speedup cpu seq to cpu task: " << to_seconds(elapsed_cpu_seq) / to_seconds(elapsed_cpu_thread) << std::endl;

	if (cpu_thread && cpu_task)
		std::cout << "speedup cpu thread to cpu task: " << to_seconds(elapsed_cpu_thread) / to_seconds(elapsed_cpu_task) << std::endl;

	if (cpu_seq && gpu)
		std::cout << "speedup cpu seq to gpu: " << to_seconds(elapsed_cpu_seq) / to_seconds(elapsed_gpu) << std::endl;

	if (cpu_thread && gpu)
		std::cout << "speedup cpu thread to gpu: " << to_seconds(elapsed_cpu_thread) / to_seconds(elapsed_gpu) << std::endl;

	if (cpu_task && gpu)
		std::cout << "speedup cpu task to gpu: " << to_seconds(elapsed_cpu_task) / to_seconds(elapsed_gpu) << std::endl;

}


/*baseline
number of pictures: 10
====== times =====
cpu seq time: 433.377 s
cpu thread time: 45.7348 s
cpu task time: 44.9392 s
gpu time: 0.695617 s
====== Speed Up =====
speedup cpu seq to cpu thread: 9.47588
speedup cpu seq to cpu task: 9.47588
speedup cpu thread to cpu task: 1.0177
speedup cpu seq to gpu: 623.011
speedup cpu thread to gpu: 65.7471
speedup cpu task to gpu: 64.6034
*/