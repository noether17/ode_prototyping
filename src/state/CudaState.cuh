#pragma once

#include <span>
#include <type_traits>
#include <utility>

// RAII class template for managing CUDA arrays.
template <typename ValueType, int N>
class CudaState {
 public:
  using StateType = std::array<ValueType, N>;
  using value_type = ValueType;
  CudaState() { cudaMalloc(&m_state, sizeof(StateType)); }
  explicit CudaState(std::span<ValueType const, N> state) {
    cudaMalloc(&m_state, sizeof(StateType));
    cudaMemcpy(m_state, state.data(), sizeof(StateType),
               cudaMemcpyHostToDevice);
  }
  CudaState(CudaState const& v) {
    cudaMalloc(&m_state, sizeof(StateType));
    cudaMemcpy(m_state, v.m_state, sizeof(StateType), cudaMemcpyDeviceToDevice);
  }
  CudaState(CudaState&& v) = default;
  auto& operator=(CudaState const& v) {
    if (this != &v) {
      cudaMemcpy(m_state, v.m_state, sizeof(StateType),
                 cudaMemcpyDeviceToDevice);
    }
    return *this;
  }
  auto& operator=(CudaState&& v) {
    std::swap(m_state, v.m_state);
    return *this;
  }
  ~CudaState() { cudaFree(m_state); }

  void copy_to(std::span<ValueType, N> copy) {
    cudaMemcpy(copy.data(), m_state, sizeof(StateType), cudaMemcpyDeviceToHost);
  }

  auto* data() { return m_state->data(); }
  auto const* data() const { return m_state->data(); }

  operator std::span<ValueType, N>() { return *m_state; }
  operator std::span<ValueType const, N>() const { return *m_state; }

  static constexpr auto size() { return N; }

 private:
  StateType* m_state{};
};

template <typename T>
inline constexpr bool IsCudaState = std::false_type{};

template <typename ValueType, int N>
inline constexpr bool IsCudaState<CudaState<ValueType, N>> = std::true_type{};

template <typename OutputStateType, typename InputStateType>
  requires(IsCudaState<InputStateType>)
auto copy_out(InputStateType const& x) {
  auto output_state = OutputStateType{};
  cudaMemcpy(output_state.data(), x.data(),
             std::size(x) * sizeof(typename InputStateType::value_type),
             cudaMemcpyDeviceToHost);
  return output_state;
}
