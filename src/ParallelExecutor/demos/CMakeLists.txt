add_executable(threadpool_n_body n_body.cpp)
target_link_libraries(threadpool_n_body
  HeapState
  NBodyODE
  RKEmbeddedParallel
  ThreadPoolExecutor
)

add_executable(cuda_n_body cuda_n_body.cu)
target_link_libraries(cuda_n_body
  CudaExecutor
  CudaState
  HeapState
  NBodyODE
  RKEmbeddedParallel
)
