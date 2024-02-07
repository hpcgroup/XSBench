cmake_minimum_required(VERSION 3.21)

project(
    XSBench_SYCL
    VERSION 1.0
    LANGUAGES CXX
)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(SOURCE Main.cpp io.cpp Simulation.cpp GridInit.cpp XSutils.cpp Materials.cpp)

add_executable(XSBench ${SOURCE})

separate_arguments(OFFLOAD_FLAGS)
target_compile_options(XSBench PUBLIC "$<$<CONFIG:RELEASE>:${OFFLOAD_FLAGS}>")
target_link_options(XSBench PUBLIC "$<$<CONFIG:RELEASE>:${OFFLOAD_FLAGS}>")

target_link_libraries(XSBench PUBLIC m)

install(TARGETS XSBench DESTINATION bin)
