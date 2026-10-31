cmake_minimum_required(VERSION 3.25)
project(woof_metal_launch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python 3.11 COMPONENTS Interpreter Development.Module REQUIRED)
if(NOT nanobind_ROOT)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT
    COMMAND_ERROR_IS_FATAL ANY)
endif()
find_package(nanobind CONFIG REQUIRED)
find_package(MLX CONFIG REQUIRED)
if(NOT MLX_BUILD_METAL)
  message(FATAL_ERROR "woof's Metal launcher needs an MLX build with Metal support")
endif()

# NB_DOMAIN mlx: share mlx.core's type registrations so mx.array crosses.
nanobind_add_module(_woof_metal_launch NB_STATIC NOMINSIZE NB_DOMAIN mlx launch.cpp)
target_link_libraries(_woof_metal_launch PRIVATE mlx)
if(WOOF_EXT_OUTPUT_DIR)
  set_target_properties(_woof_metal_launch PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${WOOF_EXT_OUTPUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${WOOF_EXT_OUTPUT_DIR}")
endif()
