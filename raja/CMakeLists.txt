cmake_minimum_required(VERSION 3.21)

project(
    XSBench_RAJA
    VERSION 1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(SOURCE Main.cpp io.cpp Simulation.cpp GridInit.cpp XSutils.cpp Materials.cpp)

add_executable(XSBench ${SOURCE})

find_package(RAJA REQUIRED)
find_package(Umpire REQUIRED)
target_link_libraries(XSBench RAJA umpire)

if (${RAJA_BACK_END} STREQUAL "CUDA")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

    set_source_files_properties(${SOURCE} PROPERTIES LANGUAGE CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -extended-lambda --expt-relaxed-constexpr --restrict --keep")
elseif (${RAJA_BACK_END} STREQUAL "HIP")
    find_package(hip REQUIRED)
    enable_language(HIP)
    set(CMAKE_HIP_STANDARD 14)
    set(CMAKE_HIP_SEPARABLE_COMPILATION ON)

    set_source_files_properties(${SOURCE} PROPERTIES LANGUAGE HIP)
endif ()

install(TARGETS XSBench DESTINATION bin)
