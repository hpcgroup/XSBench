cmake_minimum_required(VERSION 3.21)

project(
    XSBench_HIP
    VERSION 1.0
    LANGUAGES CXX HIP
)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(SOURCE Main.cpp io.cpp Simulation.cpp GridInit.cpp XSutils.cpp Materials.cpp)

set_source_files_properties(${SOURCE} PROPERTIES LANGUAGE HIP)

add_executable(XSBench ${SOURCE})

install(TARGETS XSBench DESTINATION bin)
