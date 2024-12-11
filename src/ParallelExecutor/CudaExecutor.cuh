#pragma once

#include "CudaErrorCheck.cuh"

auto static constexpr block_size = 256;

template <auto parallel_kernel, typename... Args>
__global__ void cuda_call_parallel_kernel(int n_items, Args... args) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < n_items) {
    parallel_kernel(i, args...);
    i += blockDim.x * gridDim.x;
  }
}

template <typename T, auto reduce, auto transform, typename... TransformArgs>
__global__ void cuda_transform_reduce(T* block_results, int n_items,
                                      TransformArgs... transform_args) {
  __shared__ T cache[block_size];
  auto thread_result = T{};
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto cache_index = threadIdx.x;
  while (i < n_items) {
    auto transform_result = transform(i, transform_args...);
    thread_result = reduce(thread_result, transform_result);
    i += blockDim.x * gridDim.x;
  }

  cache[cache_index] = thread_result;
  __syncthreads();
  for (auto stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (cache_index < stride) {
      cache[cache_index] =
          reduce(cache[cache_index], cache[cache_index + stride]);
    }
    __syncthreads();
  }

  if (cache_index == 0) {
    block_results[blockIdx.x] = cache[0];
  }
}

template <typename T, auto reduce>
__global__ void cuda_transform_reduce_final(T* result, T const* block_results,
                                            int n_block_results) {
  __shared__ T cache[block_size];
  auto thread_result = T{};
  auto i = threadIdx.x;  // final reduction step is always single block
  while (i < n_block_results) {
    thread_result = reduce(thread_result, block_results[i]);
    i += blockDim.x;
  }

  cache[threadIdx.x] = thread_result;
  __syncthreads();
  for (auto stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      cache[threadIdx.x] =
          reduce(cache[threadIdx.x], cache[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    *result = cache[0];
  }
}

class CudaExecutor {
  auto static constexpr block_size = 256;
  auto static constexpr n_blocks(int N) {
    return (N + block_size - 1) / block_size;
  }

 public:
  template <auto parallel_kernel, typename... Args>
  void call_parallel_kernel(int n_items, Args... args) {
    cuda_call_parallel_kernel<parallel_kernel, Args...>
        <<<n_blocks(n_items), block_size>>>(n_items, args...);
    CUDA_ERROR_CHECK(cudaGetLastError());
  }

  template <typename T, auto reduce, auto transform, typename... TransformArgs>
  auto transform_reduce(T init_val, int n_items,
                        TransformArgs... transform_args) {
    auto dev_result = (T*){nullptr};
    CUDA_ERROR_CHECK(cudaMalloc(&dev_result, sizeof(T)));
    auto dev_block_results = (T*){nullptr};
    CUDA_ERROR_CHECK(
        cudaMalloc(&dev_block_results, n_blocks(n_items) * sizeof(T)));

    cuda_transform_reduce<T, reduce, transform>
        <<<n_blocks(n_items), block_size>>>(dev_block_results, n_items,
                                            transform_args...);
    CUDA_ERROR_CHECK(cudaGetLastError());
    cuda_transform_reduce_final<T, reduce>
        <<<1, block_size>>>(dev_result, dev_block_results, n_blocks(n_items));
    CUDA_ERROR_CHECK(cudaGetLastError());

    auto result = T{};
    CUDA_ERROR_CHECK(
        cudaMemcpy(&result, dev_result, sizeof(T), cudaMemcpyDeviceToHost));

    CUDA_ERROR_CHECK(cudaFree(dev_block_results));
    CUDA_ERROR_CHECK(cudaFree(dev_result));

    return result;
  }
};