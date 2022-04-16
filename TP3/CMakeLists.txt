cmake_minimum_required(VERSION 3.10.0)
set(CMAKE_VERBOSE_MAKEFILE false)
set(PROJECT_NAME TP1)
set(TARGETS exo1 exo2 exo3 exo4 exo5)

#get info
cmake_host_system_information(RESULT MY_INFO QUERY NUMBER_OF_LOGICAL_CORES)
MESSAGE(INFO ": Number of logical cores: ${MY_INFO}")
cmake_host_system_information(RESULT MY_INFO QUERY PROCESSOR_DESCRIPTION)
MESSAGE(INFO ": processor description: ${MY_INFO}")
cmake_host_system_information(RESULT MY_INFO QUERY OS_PLATFORM)
MESSAGE(INFO ": OS platform: ${MY_INFO}")
cmake_host_system_information(RESULT SYSTEM_NAME QUERY OS_NAME)
MESSAGE(INFO ": System name: ${SYSTEM_NAME}")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# set C++
project(${PROJECT_NAME} LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)
link_libraries(Threads::Threads)

include_directories(src student utils) 

# for linux/mac but not for windows
IF(NOT "${SYSTEM_NAME}" MATCHES "Windows")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fomit-frame-pointer -O3 -Wno-ignored-attributes -mno-vzeroupper -fno-tree-vectorize")
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
    utils/Exercise.h utils/chronoCPU.hpp utils/chronoCPU.cpp utils/OPP.h utils/OPP.cpp
    student/exo3/student.h student/exo3/student.cpp student/exo3/transform.h
)
add_executable(exo4 
    src/exercise4/main.cpp src/exercise4/ExerciseImpl.cpp src/exercise4/ExerciseImpl.h
    utils/Exercise.h utils/chronoCPU.hpp utils/chronoCPU.cpp utils/OPP.h utils/OPP.cpp
    student/exo4/student.h student/exo4/student.cpp student/exo4/gather.h student/exo4/scatter.h
)
add_executable(exo5 
    src/exercise5/main.cpp src/exercise5/ExerciseImpl.cpp src/exercise5/ExerciseImpl.h
    utils/Exercise.h utils/chronoCPU.hpp utils/chronoCPU.cpp utils/OPP.h utils/OPP.cpp
    student/exo5/student.h student/exo5/student.cpp
)

# We need Thread Building Block with G++-9 or greater to activate parallelism
# If needed, installation command is "sudo apt install libtbb-dev libtbb2"
IF(NOT "${SYSTEM_NAME}" MATCHES "Windows")
target_link_libraries(exo1 "-ltbb")
target_link_libraries(exo2 "-ltbb")
target_link_libraries(exo3 "-ltbb")
target_link_libraries(exo4 "-ltbb")
target_link_libraries(exo5 "-ltbb")
endif()
