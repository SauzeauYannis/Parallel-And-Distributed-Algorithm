cmake_minimum_required(VERSION 3.10.0)
set(PROJECT_NAME TP11)
set(TARGETS Decouverte Transposition Produit Cannon Fox Snyder)

include_directories(src student . utils)
  
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_CXX_FLAGS}")  

project(${PROJECT_NAME} LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)
find_package(MPI REQUIRED)

# Exercise 1
add_executable(Decouverte 
  src/1-DecouverteTopologie/main.cc
  utils/DistributedBlockVector.h utils/DistributedBlockMatrix.h
)
# Exercise 2
add_executable(Transposition 
  src/2-Transposition/main.cc src/2-Transposition/Transposition.h 
  students/2-Transposition/Transposition.cc
  utils/DistributedBlockVector.h utils/DistributedBlockMatrix.h
)
# Exercise 3
add_executable(Produit 
  src/3-Produit/main.cc src/3-Produit/Produit.h 
  students/3-Produit/Produit.cc
  utils/DistributedBlockVector.h utils/DistributedBlockMatrix.h
  utils/chronoCPU.h utils/chronoCPU.cc
)
# Exercise 4
add_executable(Cannon 
  src/4-Cannon/main.cc src/4-Cannon/Produit.h 
  students/4-Cannon/Produit.cc
  utils/DistributedBlockVector.h utils/DistributedBlockMatrix.h
  utils/chronoCPU.h utils/chronoCPU.cc
)
# Exercise 5
add_executable(Fox 
  src/5-Fox/main.cc src/5-Fox/Produit.h 
  students/5-Fox/Produit.cc
  utils/DistributedBlockVector.h utils/DistributedBlockMatrix.h
  utils/chronoCPU.h utils/chronoCPU.cc
)
# Exercise 6
add_executable(Snyder 
  src/6-Snyder/main.cc src/6-Snyder/Produit.h 
  students/6-Snyder/Produit.cc
  utils/DistributedBlockVector.h utils/DistributedBlockMatrix.h
  utils/chronoCPU.h utils/chronoCPU.cc
)

target_link_libraries(Decouverte PUBLIC MPI::MPI_CXX Threads::Threads)
target_link_libraries(Transposition PUBLIC MPI::MPI_CXX Threads::Threads)
target_link_libraries(Produit PUBLIC MPI::MPI_CXX Threads::Threads)
target_link_libraries(Cannon PUBLIC MPI::MPI_CXX Threads::Threads)
target_link_libraries(Fox PUBLIC MPI::MPI_CXX Threads::Threads)
target_link_libraries(Snyder PUBLIC MPI::MPI_CXX Threads::Threads)

message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")
