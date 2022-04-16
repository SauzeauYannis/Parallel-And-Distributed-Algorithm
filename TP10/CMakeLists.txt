cmake_minimum_required(VERSION 3.10.0)
set(PROJECT_NAME TP10)
set(TARGETS Resolution Produit)

include_directories(src student utils)  

project(${PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_CXX_FLAGS}")  

project(${PROJECT_NAME} LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

########################################
file(GLOB OPP_MPI utils/MPI/*)
set(OPP_CHRONO utils/chronoCPU.h utils/chronoCPU.cc)

########################################
file(GLOB ResolutionSRC src/1-Resolution/* students/1-Resolution/*)
add_executable(Resolution ${ResolutionSRC} ${OPP_CHRONO} ${OPP_MPI})    

########################################
file(GLOB ProduitSRC src/2-Produit/* students/2-Produit/*)
add_executable(Produit ${ProduitSRC} ${OPP_CHRONO} ${OPP_MPI})

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
    target_link_libraries(Resolution PUBLIC MPI::MPI_CXX)
    target_link_libraries(Produit PUBLIC MPI::MPI_CXX)
endif()

message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")
