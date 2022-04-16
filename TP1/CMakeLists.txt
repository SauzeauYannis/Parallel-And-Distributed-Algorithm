cmake_minimum_required(VERSION 3.10)

MESSAGE(STATUS, "System name: ${CMAKE_SYSTEM_NAME}")

set(PROJECT_NAME TP1)
set(CMAKE_VERBOSE_MAKEFILE false)
set(TARGETS exo1 exo2 exo3 exo4 exo5)

# add CXX compiler
project(${PROJECT_NAME} LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# add threads
find_package(Threads REQUIRED)
link_libraries(Threads::Threads)

include_directories(src student utils)
IF(NOT "${CMAKE_SYSTEM_NAME}" MATCHES "Windows")
    add_definitions(-march=native -fomit-frame-pointer -O2 -Wno-ignored-attributes -mno-vzeroupper -fno-tree-vectorize)    
ENDIF()

# Need to put manually all the inputs, else MOC will fail :-(
add_executable(exo1 
    src/exercise1/main.cpp src/exercise1/Exercise1.cpp src/exercise1/Exercise1.h
    utils/Exercise.h utils/chronoCPU.hpp utils/chronoCPU.cpp
    student/exo1/student.h student/exo1/student.cpp
)

add_executable(exo2 
    src/exercise2/main.cpp src/exercise2/Exercise2.cpp src/exercise2/Exercise2.h
    utils/Exercise.h utils/chronoCPU.hpp utils/chronoCPU.cpp
    student/exo2/student.h student/exo2/student.cpp
)

add_executable(exo3 
    src/exercise3/main.cpp src/exercise3/Exercise3.cpp src/exercise3/Exercise3.h
    utils/Exercise.h utils/chronoCPU.hpp utils/chronoCPU.cpp
    student/exo3/student.h student/exo3/student.cpp
)

add_executable(exo4 
    src/exercise4/main.cpp src/exercise4/Exercise4.cpp src/exercise4/Exercise4.h
    utils/Exercise.h utils/chronoCPU.hpp utils/chronoCPU.cpp
    student/exo4/student.h student/exo4/student.cpp
)

add_executable(exo5 
    src/exercise5/main.cpp src/exercise5/Exercise5.cpp src/exercise5/Exercise5.h
    utils/Exercise.h utils/chronoCPU.hpp utils/chronoCPU.cpp
    student/exo5/student.h student/exo5/student.cpp
)
