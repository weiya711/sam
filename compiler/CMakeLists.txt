cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)
project(taco-bench)
set(CMAKE_BUILD_TYPE "release")
set(BENCHMARK_DOWNLOAD_DEPENDENCIES true)
option(NEVA "Build benchmarks for the neva or kiwi clusters." OFF)
option(LANKA "Build benchmarks for the lanka cluster." OFF)
option(BENCH_CUSPARSE_OP "BENCH_CUSPARSE_OP" OFF)

# set(ENV{MKLROOT} "/opt/intel/oneapi/mkl/2023.2.0/")
# set(ENV{MKL_INCLUDE} "${MKLROOT}/include/")
# set(ENV{MKL_LINK} "static")
# set(MKL_LINK "static")
# set(ENV{MKL_THREADING} "gnu_thread")
# set(MKL_THREADING "gnu_thread")

# set(MKL_INCLUDE "${MKLROOT}/include/")
# 
# set(MKL_LIBRARY_FLAGS -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a) 
# set(MKL_LIBRARY_FLAGS ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a) 
# set(MKL_LIBRARY_FLAGS ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp -fopenmp -openmp -lpthread -lm -ldl)
# 
# target_include_directories(taco-bench PUBLIC ${MKLROOT}/include) 

# find_package(MKL REQUIRED)
# find_package(OpenMP REQUIRED)

add_subdirectory(taco)
add_subdirectory(benchmark)
find_package(CUDA REQUIRED)

# include_directories(taco taco/include benchmark/include taco/src ${MKL_INCLUDE_DIRS} ${MKL_INCLUDE} baseline_cusparse ${CUDA_INCLUDE_DIRS})
include_directories(taco taco/include benchmark/include taco/src baseline_cusparse ${CUDA_INCLUDE_DIRS})

file(GLOB TEST_SOURCES *.cpp *.h)
file(GLOB CUSPARSE_TEST_SOURCES baseline_cusparse/*.cu bench.cpp)

# set(CMAKE_CXX_FLAGS "${C_CXX_FLAGS} -std=c++14 -openmp -fopenmp")
set(CMAKE_CXX_FLAGS "${C_CXX_FLAGS} -std=c++14")

add_executable(taco-bench ${TEST_SOURCES} bench.h)

# if(MKL_FOUND)
#     # include_directories(${MKL_INCLUDE_DIRS})
#     target_link_libraries(taco-bench "${MKL_CORE_LIBRARY}")
#     target_link_libraries(taco-bench "-Wl,--start-group" ${MKL_LIBRARIES} "-Wl,--end-group -lgomp -lpthread -lm -ldl")
# else()
#     message(WARNING "MKL libs not found")
# endif()

# if(OpenMP_CXX_FOUND)
# 	target_link_libraries(taco-bench OpenMP::OpenMP_CXX)
# endif()

target_link_libraries(taco-bench benchmark::benchmark)
target_link_libraries(taco-bench taco)

cuda_add_executable(cusparsething ${CUSPARSE_TEST_SOURCES} bench.h)
if(BENCH_CUSPARSE_OP)
  message("Using BENCHMARK version of cusparse")
  target_compile_definitions(cusparsething PRIVATE BENCH_CUSPARSE)
else()
  message("Using INTERACTIVE version of cusparse")
endif()
# I went insane
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
