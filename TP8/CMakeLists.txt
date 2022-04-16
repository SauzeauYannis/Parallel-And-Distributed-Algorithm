cmake_minimum_required(VERSION 3.0.0)
set(CMAKE_VERBOSE_MAKEFILE false)
set(PROJECT_NAME TP1)
set(TARGETS exo1 exo2 exo3 exo4)

# set CUDA
set(CUDA_STANDARD 14)
set(CUDA_STANDARD_REQUIRED ON)
set(CUDA_64_BIT_DEVICE_CODE ON)
set(CUDA_VERBOSE_BUILD  ON)

#get info
cmake_host_system_information(RESULT MY_INFO QUERY NUMBER_OF_LOGICAL_CORES)
MESSAGE(INFO ": Number of logical cores: ${MY_INFO}")
cmake_host_system_information(RESULT MY_INFO QUERY PROCESSOR_DESCRIPTION)
MESSAGE(INFO ": processor description: ${MY_INFO}")
cmake_host_system_information(RESULT MY_INFO QUERY OS_PLATFORM)
MESSAGE(INFO ": OS platform: ${MY_INFO}")
cmake_host_system_information(RESULT SYSTEM_NAME QUERY OS_NAME)
MESSAGE(INFO ": System name: ${SYSTEM_NAME}")

# set C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#if(NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES})
#    set(CMAKE_CUDA_ARCHITECTURES 52)
#endif()
message(STATUS "CUDA architectures set to ${CMAKE_CUDA_ARCHITECTURES}")

# add CXX/CUDA compiler
project(${PROJECT_NAME} LANGUAGES CXX CUDA)

include_directories(src student utils ${CUDA_SAMPLES_INC})

# OPENMP for linux/mac but not for windows
IF(NOT "${SYSTEM_NAME}" MATCHES "Windows")
    set (CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS} -march=native -fomit-frame-pointer -O3 -mavx -ffast-math -funroll-loops -ftree-vectorize -floop-parallelize-all")
    set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 ")
ELSE()
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ")
ENDIF()

# Need to put manually all the inputs, else MOC will fail :-(
add_executable(exo1
    src/exercise1/main.cu src/exercise1/ExerciseImpl.h src/exercise1/ExerciseImpl.cu
    src/reference/ImageEqualizer.h src/reference/ImageEqualizer.cpp
    student/exo1/student.h student/exo1/student.cu
    utils/chronoGPU.hpp utils/chronoGPU.cu utils/OPP.h utils/chronoCPU.cpp utils/chronoCPU.hpp
    utils/ppm.h utils/ppm.cpp
)
add_executable(exo2
    src/exercise2/main.cu src/exercise2/ExerciseImpl.h src/exercise2/ExerciseImpl.cu
    src/reference/ImageEqualizer.h src/reference/ImageEqualizer.cpp
    student/exo2/student.h student/exo2/student.cu
    utils/chronoGPU.hpp utils/chronoGPU.cu utils/OPP.h utils/chronoCPU.cpp utils/chronoCPU.hpp
    utils/ppm.h utils/ppm.cpp
)
add_executable(exo3
    src/exercise3/main.cu src/exercise3/ExerciseImpl.h src/exercise3/ExerciseImpl.cu
    src/reference/ImageEqualizer.h src/reference/ImageEqualizer.cpp
    student/exo3/student.h student/exo3/student.cu
    utils/chronoGPU.hpp utils/chronoGPU.cu utils/OPP.h utils/chronoCPU.cpp utils/chronoCPU.hpp
    utils/ppm.h utils/ppm.cpp
)
add_executable(exo4
    src/exercise4/main.cu src/exercise4/ExerciseImpl.h src/exercise4/ExerciseImpl.cu
    src/reference/ImageEqualizer.h src/reference/ImageEqualizer.cpp
    student/exo4/student.h student/exo4/student.cu
    utils/chronoGPU.hpp utils/chronoGPU.cu utils/OPP.h utils/chronoCPU.cpp utils/chronoCPU.hpp
    utils/ppm.h utils/ppm.cpp
)

IF(${CMAKE_VERSION} VERSION_LESS "3.18.0")
    target_compile_options(exo1 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -gencode=arch=compute_37,code=sm_37;
        -gencode=arch=compute_50,code=sm_50;
        -gencode=arch=compute_60,code=sm_60;
        -gencode=arch=compute_37,code=sm_37;
        -gencode=arch=compute_70,code=sm_70;
        -gencode=arch=compute_75,code=sm_75;
        -gencode=arch=compute_80,code=sm_80>)
    target_compile_options(exo2 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -gencode=arch=compute_37,code=sm_37;
        -gencode=arch=compute_50,code=sm_50;
        -gencode=arch=compute_60,code=sm_60;
        -gencode=arch=compute_37,code=sm_37;
        -gencode=arch=compute_70,code=sm_70;
        -gencode=arch=compute_75,code=sm_75;
        -gencode=arch=compute_80,code=sm_80>)
    target_compile_options(exo3 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -gencode=arch=compute_37,code=sm_37;
        -gencode=arch=compute_50,code=sm_50;
        -gencode=arch=compute_60,code=sm_60;
        -gencode=arch=compute_37,code=sm_37;
        -gencode=arch=compute_70,code=sm_70;
        -gencode=arch=compute_75,code=sm_75;
        -gencode=arch=compute_80,code=sm_80>)
    target_compile_options(exo4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -gencode=arch=compute_37,code=sm_37;
        -gencode=arch=compute_50,code=sm_50;
        -gencode=arch=compute_60,code=sm_60;
        -gencode=arch=compute_37,code=sm_37;
        -gencode=arch=compute_70,code=sm_70;
        -gencode=arch=compute_75,code=sm_75;
        -gencode=arch=compute_80,code=sm_80>)
ELSE()
    set_property(TARGET exo1 PROPERTY CUDA_ARCHITECTURES 37 52 60 61 75)
    set_property(TARGET exo2 PROPERTY CUDA_ARCHITECTURES 37 52 60 61 75)
    set_property(TARGET exo3 PROPERTY CUDA_ARCHITECTURES 37 52 60 61 75)
    set_property(TARGET exo4 PROPERTY CUDA_ARCHITECTURES 37 52 60 61 75)
ENDIF()