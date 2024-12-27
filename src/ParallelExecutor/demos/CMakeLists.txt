add_executable(parallel_n_body n_body.cpp)
target_link_libraries(parallel_n_body
  HeapState
  ParallelThreadPool
  RKEmbeddedParallel
)

add_executable(cuda_n_body cuda_n_body.cu)
target_link_libraries(cuda_n_body
  CudaExecutor
  CudaState
  RKEmbeddedParallel
)
