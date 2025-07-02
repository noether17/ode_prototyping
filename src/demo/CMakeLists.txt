add_subdirectory(n_body_scenarios)

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

add_executable(cuda_n_body_leapfrog cuda_n_body_leapfrog.cu)
target_link_libraries(cuda_n_body_leapfrog
  CudaExecutor
  CudaState
  HeapState
  LeapfrogParallel
  NBodyODE
  RawOutput
)

add_executable(cuda_n_body_box cuda_n_body_box.cu)
target_link_libraries(cuda_n_body_box
  BTRKF78
  CudaExecutor
  CudaState
  HeapState
  NBodyODE
  ParticlesInBox
  RawOutput
  RKEmbeddedParallel
)

add_executable(n_body_benchmark_test n_body_benchmark_test.cu)
target_link_libraries(n_body_benchmark_test
  AtomicUtil
  BTRKF78
  CudaExecutor
  CudaState
  HeapState
  NBodySimpleODE
  ParticlesInBox
  RawOutput
  RKEmbeddedParallel
)
