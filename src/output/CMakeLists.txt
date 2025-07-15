add_library(RawOutput INTERFACE RawOutput.hpp)
target_include_directories(RawOutput INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(RawOutputWithLog INTERFACE RawOutputWithLog.hpp)
target_include_directories(RawOutputWithLog INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
