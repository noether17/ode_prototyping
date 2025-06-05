#pragma once

#include <span>
#include <type_traits>
#include <utility>

// RAII class template for managing CUDA arrays.
template <template <typename, std::size_t> typename ContainerType, typename T,
          std::size_t N>
class CudaState {
 public:
  template <typename ValueType, std::size_t Size>
  using container_type = ContainerType<ValueType, Size>;
  using state_type = container_type<T, N>;
  using value_type = T;

  CudaState() { cudaMalloc(&m_state, sizeof(state_type)); }
  explicit CudaState(state_type const& state) {
    cudaMalloc(&m_state, sizeof(state_type));
    cudaMemcpy(m_state, state.data(), sizeof(state_type),
               cudaMemcpyHostToDevice);
  }
  explicit CudaState(std::span<value_type const, N> state) {
    cudaMalloc(&m_state, sizeof(state_type));
    cudaMemcpy(m_state, state.data(), sizeof(state_type),
               cudaMemcpyHostToDevice);
  }
  CudaState(CudaState const& v) {
    cudaMalloc(&m_state, sizeof(state_type));
    cudaMemcpy(m_state, v.m_state, sizeof(state_type),
               cudaMemcpyDeviceToDevice);
  }
  CudaState(CudaState&& v) = default;
  auto& operator=(CudaState const& v) {
    if (this != &v) {
      cudaMemcpy(m_state, v.m_state, sizeof(state_type),
                 cudaMemcpyDeviceToDevice);
    }
    return *this;
  }
  auto& operator=(CudaState&& v) {
    std::swap(m_state, v.m_state);
    return *this;
  }
  ~CudaState() { cudaFree(m_state); }

  void copy_to(std::span<value_type, N> copy) {
    cudaMemcpy(copy.data(), m_state, sizeof(state_type),
               cudaMemcpyDeviceToHost);
  }

  auto* data() { return m_state->data(); }
  auto const* data() const { return m_state->data(); }

  operator std::span<value_type, N>() { return *m_state; }
  operator std::span<value_type const, N>() const { return *m_state; }

  static constexpr auto size() { return N; }

 private:
  state_type* m_state{};
};

template <typename T>
inline constexpr bool IsCudaState = std::false_type{};

template <template <typename, std::size_t> typename ContainerType, typename T,
          std::size_t N>
inline constexpr bool IsCudaState<CudaState<ContainerType, T, N>> =
    std::true_type{};

template <typename OutputStateType, typename InputStateType>
  requires(IsCudaState<InputStateType>)
auto copy_out(InputStateType const& x) {
  auto output_state = OutputStateType{};
  cudaMemcpy(output_state.data(), x.data(),
             std::size(x) * sizeof(typename InputStateType::value_type),
             cudaMemcpyDeviceToHost);
  return output_state;
}
