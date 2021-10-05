set(BINARY ${CMAKE_PROJECT_NAME})
set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_library(${CMAKE_PROJECT_NAME}_lib STATIC
  main.cpp
  data.cpp
  BayesRRm.cpp
  BayesW_arms.cpp
  BayesW.cpp
  gadgets.cpp
  distributions_boost.cpp
  options.cpp
  samplewriter.cpp
  hydra.h
)

add_executable(${BINARY} main.cpp)

target_link_libraries(${BINARY} ${CMAKE_PROJECT_NAME}_lib MPI::MPI_CXX  OpenMP::OpenMP_CXX)
if(TARGET mtrait_lib)
  target_link_libraries(${BINARY} mtrait_lib)
endif()

Set(${CMAKE_PROJECT_NAME}_INCLUDE_DIRS ${CMMAKE_PROJECT_SOURCE_DIR} ${CMAKE_PROJECT_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR} CACHE INTERNAL "${CMAKE_PROJECT_NAME}: Include Directories"
    FORCE)
