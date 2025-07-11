add_executable(n_body_benchmark n_body_benchmark.cu)
target_link_libraries(n_body_benchmark
  benchmark::benchmark
  benchmark::benchmark_main
  BTHE21
  BTRKF45
  BTDOPRI5
  BTDVERK
  BTRKF78
  CudaExecutor
  CudaState
  HeapState
  NBodyODE
  RKEmbeddedParallel
  SingleThreadedExecutor
  SpinningParticlesInBox
  ThreadPoolExecutor
)

