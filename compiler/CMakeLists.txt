cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)
project(taco-bench)
set(CMAKE_BUILD_TYPE "release")
set(BENCHMARK_DOWNLOAD_DEPENDENCIES true)
option(NEVA "Build benchmarks for the neva or kiwi clusters." OFF)
option(LANKA "Build benchmarks for the lanka cluster." OFF)

add_subdirectory(taco)
add_subdirectory(benchmark)
find_package(CUDA REQUIRED)

include_directories(taco taco/include benchmark/include taco/src baseline_cusparse ${CUDA_INCLUDE_DIRS})

file(GLOB TEST_SOURCES *.cpp *.h)
file(GLOB CUSPARSE_TEST_SOURCES baseline_cusparse/*.cu baseline.cpp bench.cpp)

set(CMAKE_CXX_FLAGS "${C_CXX_FLAGS} -std=c++14")

add_executable(taco-bench ${TEST_SOURCES} bench.h)
target_link_libraries(taco-bench benchmark::benchmark)
target_link_libraries(taco-bench taco)

# I went insane
cuda_add_executable(cusparsething ${CUSPARSE_TEST_SOURCES} bench.h)
target_link_libraries(cusparsething /usr/local/cuda-12.2/lib64/libcusparse_static.a)
target_link_libraries(cusparsething taco)
target_link_libraries(cusparsething cudart)
target_link_libraries(cusparsething benchmark::benchmark)


# Only link stdc++fs on linux. Mac's don't need this set.
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(MACOSX TRUE)
endif()
if (NOT MACOSX)
  target_link_libraries(taco-bench stdc++fs)
endif()
