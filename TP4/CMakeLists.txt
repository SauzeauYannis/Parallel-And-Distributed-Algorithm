cmake_minimum_required(VERSION 3.0.0)
set(CMAKE_VERBOSE_MAKEFILE false)
set(PROJECT_NAME TP1)
set(TARGETS exo1 exo2 exo3)

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

# add CXX/CUDA compiler
project(${PROJECT_NAME} LANGUAGES CXX) #CUDA)

include_directories(src student utils) #${CUDA_SAMPLES_INC})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_CXX_FLAGS} -I./src -I./student -I./utils")

# for linux/mac but not for windows
IF(NOT "${SYSTEM_NAME}" MATCHES "Windows")
    set (CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS} -march=native -fomit-frame-pointer -O2 -Wno-ignored-attributes -mno-vzeroupper -fno-tree-vectorize")
    find_package(OpenMP REQUIRED)
    if (OPENMP_FOUND)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
ELSE()
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}") # /O2
ENDIF()

# Need to put manually all the inputs, else MOC will fail :-(
add_executable(exo1 
    src/exercise1/main.cpp src/exercise1/ExerciseImpl.cpp src/exercise1/ExerciseImpl.h
    utils/Exercise.h utils/chronoCPU.hpp utils/chronoCPU.cpp utils/iMatrix22.h utils/iMatrix22.cpp
    student/exo1/student.h student/exo1/inclusive_scan.h utils/OPP.h utils/OPP.cpp
)

add_executable(exo2 
    src/exercise2/main.cpp src/exercise2/ExerciseImpl.cpp src/exercise2/ExerciseImpl.h
    utils/Exercise.h utils/chronoCPU.hpp utils/chronoCPU.cpp utils/iMatrix22.h utils/iMatrix22.cpp
    student/exo2/student.h student/exo2/exclusive_scan.h utils/OPP.h utils/OPP.cpp
)


