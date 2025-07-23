cmake_minimum_required(VERSION 3.22...4.0)
project(LearnedStaticFunction)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif ()

if((CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        AND PROJECT_IS_TOP_LEVEL)
    add_compile_options(-march=native -ffast-math)
endif()

####################### Dependency setup #######################

set(PSIMD_SOURCE_DIR "../../lib/psimd")
#option(TFLITE_ENABLE_METAL "Enable Metal delegate (iOS only)" ON)
#option(TFLITE_ENABLE_XNNPACK "Enable XNNPACK backend" OFF)
add_subdirectory(lib/tensorflow/tensorflow/lite "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite" EXCLUDE_FROM_ALL)

if(NOT TARGET tlx)
    set(TLX_INSTALL_INCLUDE_DIR tlx CACHE PATH "Workaround for TLX breaking the first cmake call")
    add_subdirectory(lib/burr-vl/tlx SYSTEM EXCLUDE_FROM_ALL)
endif()

if(NOT TARGET ips2ra)
    set(IPS2RA_DISABLE_PARALLEL ON CACHE PATH "FindTBB greps a file that does not exist in recent TBB versions")
    add_subdirectory(lib/burr-vl/ips2ra SYSTEM EXCLUDE_FROM_ALL)
endif()

add_library(RibbonSorter SHARED lib/burr-vl/sorter.cpp)
target_compile_features(RibbonSorter PUBLIC cxx_std_17)
target_link_libraries(RibbonSorter PUBLIC tlx ips2ra)

add_library(RibbonVLR INTERFACE)
target_compile_features(RibbonVLR INTERFACE cxx_std_20)
target_include_directories(RibbonVLR INTERFACE lib/burr-vl)
target_include_directories(RibbonVLR INTERFACE lib/burr-vl/ips2ra/include)
target_include_directories(RibbonVLR INTERFACE lib/burr-vl/DySECT)
target_include_directories(RibbonVLR INTERFACE lib/sort/include)
target_link_libraries(RibbonVLR INTERFACE RibbonSorter)

####################### Library Setup #######################

add_library(LearnedStaticFunction INTERFACE)
target_include_directories(LearnedStaticFunction INTERFACE include)
target_compile_features(LearnedStaticFunction INTERFACE cxx_std_23)
target_link_libraries(LearnedStaticFunction INTERFACE RibbonVLR tensorflow-lite)

####################### Benchmark Targets #######################
if(PROJECT_IS_TOP_LEVEL)
    add_executable(ribbon_learned_bench ribbon_learned_bench.cpp)
    target_link_libraries(ribbon_learned_bench PUBLIC LearnedStaticFunction tlx)

    add_executable(plot_model_calibration plot_model_calibration.cpp)
    target_link_libraries(plot_model_calibration PUBLIC LearnedStaticFunction tlx)

    add_executable(filter_tuner filter_tuner.cpp)
endif()
