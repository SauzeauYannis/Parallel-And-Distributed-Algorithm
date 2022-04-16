cmake_minimum_required(VERSION 3.10.0)
set(PROJECT_NAME Cours)
set(TARGETS PremierPas Communication NaiveBroadcast PipeBroadcast)

include_directories(src student utils)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_CXX_FLAGS}")  

project(${PROJECT_NAME} LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

#set(OPP_MPI utils/MPI/OPP_MPI.h utils/MPI/OPP_MPI_base.h utils/MPI/OPP_MPI_Ring.h utils/MPI/OPP_MPI_Torus.h)
file(GLOB OPP_MPI utils/MPI/*)
set(OPP_CHRONO utils/chronoCPU.h utils/chronoCPU.cc)

add_executable(PremierPas src/1-PremierPas/main.cc)

add_executable(Communication students/2-Communication/main.cc ${OPP_MPI}) 

file(GLOB NaiveSRC src/3-1-broadcast/* students/3-1-Broadcast/*)
add_executable(NaiveBroadcast ${NaiveSRC} ${OPP_CHRONO} ${OPP_MPI})    

file(GLOB PipeSRC src/3-2-broadcast/* students/3-2-Broadcast/*)
add_executable(PipeBroadcast ${PipeSRC} ${OPP_CHRONO} ${OPP_MPI})

find_package(MPI REQUIRED)

# For supporting CMake < 3.9:
if(NOT TARGET MPI::MPI_CXX)
    message(STATUS "Old MPI behavior ...")
    add_library(MPI::MPI_CXX IMPORTED INTERFACE)    
    set_property(TARGET MPI::MPI_CXX
                 PROPERTY INTERFACE_COMPILE_OPTIONS ${MPI_CXX_COMPILE_FLAGS})
    set_property(TARGET MPI::MPI_CXX
                 PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${MPI_CXX_INCLUDE_PATH}")
    set_property(TARGET MPI::MPI_CXX
                 PROPERTY INTERFACE_LINK_LIBRARIES ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES})
else()
    message(STATUS "New MPI behavior :-)")
    target_link_libraries(PremierPas PUBLIC MPI::MPI_CXX)
    target_link_libraries(Communication PUBLIC MPI::MPI_CXX)
    target_link_libraries(NaiveBroadcast PUBLIC MPI::MPI_CXX)
    target_link_libraries(PipeBroadcast PUBLIC MPI::MPI_CXX)
endif()

message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")