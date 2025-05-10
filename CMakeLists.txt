cmake_minimum_required(VERSION 3.22)
project(LearnedRetrieval)

set(CMAKE_CXX_STANDARD 23)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

include_directories(include lib/sort/include)

#option(TFLITE_ENABLE_METAL "Enable Metal delegate (iOS only)" ON)
#option(TFLITE_ENABLE_XNNPACK "Enable XNNPACK backend" OFF)
add_subdirectory(lib/tensorflow/tensorflow/lite "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite" EXCLUDE_FROM_ALL)

add_executable(LearnedRetrieval main.cpp)
target_link_libraries(LearnedRetrieval tensorflow-lite)
