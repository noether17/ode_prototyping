add_executable(threadpool_n_body n_body.cpp)
target_link_libraries(threadpool_n_body
  BTRKF78
  HeapState
  NBodyODE
  RawOutput
  RKEmbeddedParallel
  ThreadPoolExecutor
)

add_executable(cuda_n_body cuda_n_body.cu)
target_link_libraries(cuda_n_body
  BTRKF78
  CudaExecutor
  CudaState
  HeapState
  NBodyODE
  RawOutput
  RKEmbeddedParallel
)
