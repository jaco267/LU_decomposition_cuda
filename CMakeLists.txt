# CMakeLists.txt  #sudo update-alternatives --config c++ #change from g++ to clang
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(main LANGUAGES CXX CUDA)  # <-- enable CUDA

find_package(Boost COMPONENTS system thread iostreams)
find_package(CUDAToolkit REQUIRED)
# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wsign-compare")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# add_subdirectory(numcpp)
# add_subdirectory(examples)
# Collect all .cpp files in the project directory (not include sub dir)
file(GLOB SOURCES "*.cpp" "*.cu")

# Add the executable with all collected source files
add_executable(${PROJECT_NAME} ${SOURCES})

# Example if you want separate flags for CUDA
# (nvcc uses --extended-lambda for C++17 lambdas by default)
# set_target_properties(${PROJECT_NAME}
#     PROPERTIES
#     CXX_STANDARD 17
#     CUDA_SEPARABLE_COMPILATION ON
#     CUDA_ARCHITECTURES 70 # change depending on GPU (e.g. 50, 61, 75, 86, 89, 90)
# )
# target_link_libraries(
#     ${PROJECT_NAME} 
#     numcpp
#     examples
# )
# Link cuBLAS
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        CUDA::cublas
)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)