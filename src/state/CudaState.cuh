#pragma once

#include <concepts>
#include <span>
#include <type_traits>
#include <utility>

#include "CudaErrorCheck.cuh"

template <std::floating_point T, std::size_t N>
class CudaState {
 public:
  using value_type = T;
  static constexpr auto size() { return N; }

  CudaState() {
    CUDA_ERROR_CHECK(cudaMalloc(&device_ptr_, N * sizeof(T)));
    CUDA_ERROR_CHECK(cudaMemset(device_ptr_, 0, N * sizeof(T)));
  }
  explicit CudaState(std::span<const T, N> state_span) {
    CUDA_ERROR_CHECK(cudaMalloc(&device_ptr_, N * sizeof(T)));
    CUDA_ERROR_CHECK(cudaMemcpy(device_ptr_, state_span.data(), N * sizeof(T),
                                cudaMemcpyHostToDevice));
  }
  CudaState(CudaState const& other) {
    CUDA_ERROR_CHECK(cudaMalloc(&device_ptr_, N * sizeof(T)));
    CUDA_ERROR_CHECK(cudaMemcpy(device_ptr_, other.device_ptr_, N * sizeof(T),
                                cudaMemcpyDeviceToDevice));
  }
  CudaState(CudaState&& other) noexcept : device_ptr_{other.device_ptr_} {
    CUDA_ERROR_CHECK(cudaMalloc(&other.device_ptr_, N * sizeof(T)));
  }
  CudaState& operator=(CudaState const& other) {
    if (this != &other) {
      CUDA_ERROR_CHECK(cudaMemcpy(device_ptr_, other.device_ptr_, N * sizeof(T),
                                  cudaMemcpyDeviceToDevice));
    }
    return *this;
  }
  CudaState& operator=(CudaState&& other) noexcept {
    std::swap(device_ptr_, other.device_ptr_);
    return *this;
  }
  ~CudaState() { cudaFree(device_ptr_); }

  auto* data() { return device_ptr_; }
  auto const* data() const { return device_ptr_; }

  operator std::span<T, N>() { return std::span<T, N>{data(), size()}; }
  operator std::span<T const, N>() const {
    return std::span<T const, N>{data(), size()};
  }
  void copy_to(std::span<value_type, N> copy) {
    cudaMemcpy(copy.data(), device_ptr_, N * sizeof(T), cudaMemcpyDeviceToHost);
  }

 private:
  T* device_ptr_{};
};

template <typename R>
CudaState(R&&) -> CudaState<typename std::remove_reference_t<R>::value_type,
                            std::tuple_size_v<std::remove_reference_t<R>>>;

template <typename T>
inline constexpr bool IsCudaState = std::false_type{};

template <typename T, std::size_t N>
inline constexpr bool IsCudaState<CudaState<T, N>> = std::true_type{};

template <typename OutputStateType, typename InputStateType>
  requires(IsCudaState<InputStateType>)
auto copy_out(InputStateType const& x) {
  auto output_state = OutputStateType{};
  cudaMemcpy(output_state.data(), x.data(),
             std::size(x) * sizeof(typename InputStateType::value_type),
             cudaMemcpyDeviceToHost);
  return output_state;
}
