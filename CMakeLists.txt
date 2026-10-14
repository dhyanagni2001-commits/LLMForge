# Optional native build. The primary build path is torch JIT (kernels/__init__.py).
cmake_minimum_required(VERSION 3.18)
project(llmforge_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
execute_process(
  COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX})
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

add_library(llmforge_cuda MODULE
  bindings/torch_extension.cpp
  kernels/cuda/fused_add_rmsnorm.cu)
target_include_directories(llmforge_cuda PRIVATE kernels/cuda ${Python_INCLUDE_DIRS})
target_link_libraries(llmforge_cuda PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
target_compile_definitions(llmforge_cuda PRIVATE TORCH_EXTENSION_NAME=llmforge_cuda)
set_target_properties(llmforge_cuda PROPERTIES PREFIX "" CUDA_ARCHITECTURES "70;75;80;86;89;90")
