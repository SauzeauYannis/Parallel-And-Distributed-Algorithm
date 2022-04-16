cmake_minimum_required(VERSION 3.0.0)
set(PROJECT_NAME TP1)
set(TARGETS exo1 exo2 exo3 exo4 exo5)

#get info
cmake_host_system_information(RESULT MY_INFO QUERY NUMBER_OF_LOGICAL_CORES)
MESSAGE(INFO ": Number of logical cores: ${MY_INFO}")
cmake_host_system_information(RESULT MY_INFO QUERY PROCESSOR_NAME)
MESSAGE(INFO ": processor name: ${MY_INFO}")
MESSAGE(STATUS, "System name: ${CMAKE_SYSTEM_NAME}")

# set C++
set(CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED ON)

# add CXX/CUDA compiler
project(${PROJECT_NAME} LANGUAGES CXX)

include_directories(src student utils)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_CXX_FLAGS} -I./src -I./student -I./utils")

# for linux/mac but not for windows
IF(NOT "${CMAKE_SYSTEM_NAME}" MATCHES "Windows")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mavx2 -fomit-frame-pointer -O3 -Wno-ignored-attributes -mno-vzeroupper -std=c++17 -fno-tree-vectorize")
    find_package(OpenMP)
    if (OPENMP_FOUND)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
ELSE()
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2 /std:c++latest")
ENDIF()

# Need to put manually all the inputs, else MOC will fail :-(
add_executable(exo1 
    src/exercise1/main.cpp src/exercise1/ExerciseImpl.cpp src/exercise1/ExerciseImpl.h
    utils/Exercise.h utils/chronoCPU.hpp utils/chronoCPU.cpp
    student/exo1/student.h student/exo1/student.cpp
)
add_executable(exo2 
    src/exercise2/main.cpp src/exercise2/ExerciseImpl.cpp src/exercise2/ExerciseImpl.h
    utils/Exercise.h utils/chronoCPU.hpp utils/chronoCPU.cpp
    student/exo2/student.h student/exo2/student.cpp
)
add_executable(exo3 
    src/exercise3/main.cpp src/exercise3/ExerciseImpl.cpp src/exercise3/ExerciseImpl.h
    utils/Exercise.h utils/chronoCPU.hpp utils/chronoCPU.cpp
    student/exo3/student.h student/exo3/student.cpp
)
add_executable(exo4 
    src/exercise4/main.cpp src/exercise4/ExerciseImpl.cpp src/exercise4/ExerciseImpl.h
    utils/Exercise.h utils/chronoCPU.hpp utils/chronoCPU.cpp
    student/exo4/student.h student/exo4/student.cpp
)
add_executable(exo5 
    src/exercise5/main.cpp src/exercise5/ExerciseImpl.cpp src/exercise5/ExerciseImpl.h
    utils/Exercise.h utils/chronoCPU.hpp utils/chronoCPU.cpp
    student/exo5/student.h student/exo5/student.cpp
)


