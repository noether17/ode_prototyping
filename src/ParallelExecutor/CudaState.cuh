#pragma once

#include <span>
#include <utility>

// RAII class template for managing CUDA arrays.
template <typename ValueType, int N>
class CudaState {
 public:
  using StateType = std::array<ValueType, N>;
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

  auto static constexpr size() { return N; }

  // TODO: This likely won't work for non-zero values.
  friend void fill(CudaState& cs, ValueType value) {
    cudaMemset(cs.m_state, value, N);
  }

 private:
  StateType* m_state{};
};
