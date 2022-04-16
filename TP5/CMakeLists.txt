cmake_minimum_required(VERSION 3.0.0)
set(CMAKE_VERBOSE_MAKEFILE false)
set(PROJECT_NAME TP1)
set(TARGETS exo1 exo2 exo3 exo4 exo5)

# set CUDA
#set(CUDA_STANDARD 11)
#set(CUDA_STANDARD_REQUIRED OFF)
#set(CUDA_64_BIT_DEVICE_CODE ON)
#set(CUDA_VERBOSE_BUILD  ON)

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

# add CXX compiler
project(${PROJECT_NAME} LANGUAGES CXX) 

find_package(Threads REQUIRED)

include_directories(src student utils) 

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_CXX_FLAGS}")

IF(NOT "${SYSTEM_NAME}" MATCHES "Windows")
    set (CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS} -march=native -fomit-frame-pointer -O2 -Wno-ignored-attributes -mno-vzeroupper -fno-tree-vectorize")
ELSE()
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}") # /O2
ENDIF()

# Need to put manually all the inputs, else MOC will fail :-(
add_executable(exo1 
    src/exercise1/main.cpp src/exercise1/ExerciseImpl.cpp src/exercise1/ExerciseImpl.h
    utils/Exercise.h utils/chronoCPU.hpp utils/chronoCPU.cpp utils/OPP.h utils/OPP.cpp
    student/exo1/student.h student/exo1/student.cpp 
)
target_link_libraries(exo1 Threads::Threads)

add_executable(exo2 
    src/exercise2/main.cpp src/exercise2/ExerciseImpl.cpp src/exercise2/ExerciseImpl.h
    utils/Exercise.h utils/chronoCPU.hpp utils/chronoCPU.cpp utils/OPP.h utils/OPP.cpp
    student/exo2/student.h student/exo2/student.cpp
)
target_link_libraries(exo2 Threads::Threads)

add_executable(exo3 
    src/exercise3/main.cpp src/exercise3/ExerciseImpl.cpp src/exercise3/ExerciseImpl.h
    utils/Exercise.h utils/chronoCPU.hpp utils/chronoCPU.cpp utils/OPP.h utils/OPP.cpp
    student/exo3/student.h student/exo3/student.cpp student/previous/transform.h student/previous/scatter.h
    student/previous/inclusive_scan.h student/previous/exclusive_scan.h
)
target_link_libraries(exo3 Threads::Threads)

add_executable(exo4 
    src/exercise4/main.cpp src/exercise4/ExerciseImpl.cpp src/exercise4/ExerciseImpl.h
    utils/Exercise.h utils/chronoCPU.hpp utils/chronoCPU.cpp utils/OPP.h utils/OPP.cpp
    student/exo4/student.h student/exo4/student.cpp student/exo4/partition.h
)
target_link_libraries(exo4 Threads::Threads)

add_executable(exo5 
    src/exercise5/main.cpp src/exercise5/ExerciseImpl.cpp src/exercise5/ExerciseImpl.h
    utils/Exercise.h utils/chronoCPU.hpp utils/chronoCPU.cpp utils/OPP.h utils/OPP.cpp
    student/exo5/student.h student/exo5/student.cpp student/previous/transform.h 
    student/exo4/partition.h
)
target_link_libraries(exo5 Threads::Threads)
