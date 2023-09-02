cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)
project(taco-bench)
set(CMAKE_BUILD_TYPE "Release")
option(NEVA "Build benchmarks for the neva or kiwi clusters." OFF)
option(LANKA "Build benchmarks for the lanka cluster." OFF)

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


add_subdirectory(taco)
add_subdirectory(benchmark)

include_directories(taco taco/include benchmark/include taco/src) 
# include_directories(${MKL_INCLUDE_DIRS} ${MKL_INCLUDE})

file(GLOB TEST_SOURCES *.cpp *.h)

set(CMAKE_CXX_FLAGS "${C_CXX_FLAGS} -std=c++14")
# find_package(MKL REQUIRED)

add_executable(taco-bench ${TEST_SOURCES} bench.h)

if(MKL_FOUND)
    # include_directories(${MKL_INCLUDE_DIRS})
    target_link_libraries(taco-bench "${MKL_CORE_LIBRARY}")
    target_link_libraries(taco-bench "-Wl,--start-group" ${MKL_LIBRARIES} "-Wl,--end-group -lgomp -lpthread -lm -ldl")
else()
    message(WARNING "MKL libs not found")
endif()

if(OpenMP_CXX_FOUND)
	target_link_libraries(taco-bench OpenMP::OpenMP_CXX)
endif()

target_link_libraries(taco-bench benchmark::benchmark)
target_link_libraries(taco-bench taco)
# Only link stdc++fs on linux. Mac's don't need this set.
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(MACOSX TRUE)
endif()
if (NOT MACOSX)
  target_link_libraries(taco-bench stdc++fs)
endif()
