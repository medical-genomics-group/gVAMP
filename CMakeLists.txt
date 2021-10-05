cmake_minimum_required(VERSION 3.10)
include(CheckCXXCompilerFlag)
project(hydra)

option(ENABLE_UNIT_TESTS "Enable unit tests" OFF)
option(AVX512 "Enable more compiler optimizations" ON)
option(NATIVE "Enable more compiler optimizations" OFF)


add_definitions("-DNEWTONIAN")





set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")
set(CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG")

#first we look for eigen in root directory, if hot found we use the system version

find_package(MPI REQUIRED)
message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")
include_directories(${MPI_INCLUDE_PATH})


find_package (Eigen3)

SET(GCC_EIGEN_COMPILE_FLAGS "-Wno-deprecated-declarations -DEIGEN_NO_DEBUG -DEIGEN_UNROLLING_LIMIT=1000")

find_package(Boost REQUIRED)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
  add_definitions("-DBOOST_ALLOW_DEPRECATED_HEADERS")
endif()


# Check for -march=native support in the compiler
if(NATIVE)
  CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
  if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
  endif()
endif()

#check for AVX support 
if(AVX512)
  Check_CXX_COMPILER_FLAG("-xCORE-AVX512" COMPILER_SUPPORTS_AVX512)
  CHECK_CXX_COMPILER_FLAG("-qopt-zmm-usage=high" COMPILER_SUPPORTS_ZZMH)
  if(COMPILER_SUPPORTS_AVX512 AND COMPILER_SUPPORTS_ZZMH)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xCORE-AVX512 -qopt-zmm-usage=high")
  endif()
endif()

# We look for boost used for random number generator
find_package(Boost REQUIRED COMPONENTS locale)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIR})
    message("-- Boost found: ${Boost_VERSION_STRING}")
else()
    message(FATAL_ERROR "Boost not found!")
endif()


find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  message("-- OpenMP found")
else()
  message(FATAL_ERROR "OpenMP not found!")
endif()

find_package(ZLIB REQUIRED)
if(ZLIB_FOUND)
  include_directories(${ZLIB_INCLUDE_DIRS})
endif(ZLIB_FOUND)




SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_EIGEN_COMPILE_FLAGS}")



include_directories(src)
add_subdirectory(src)


if(ENABLE_UNIT_TESTS)
  
  # Download and unpack googletest at configure time
  configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
  execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download"
    )
  execute_process(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download"
    )

  # Prevent GoogleTest from overriding our compiler/linker options
  # when building with Visual Studio
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

  # Add googletest directly to our build. This adds the following targets:
  # gtest, gtest_main, gmock and gmock_main
  add_subdirectory("${CMAKE_BINARY_DIR}/googletest-src"
    "${CMAKE_BINARY_DIR}/googletest-build"
    )

  # The gtest/gmock targets carry header search path dependencies
  # automatically when using CMake 2.8.11 or later. Otherwise we
  # have to add them here ourselves.
  if(CMAKE_VERSION VERSION_LESS 2.8.11)
    include_directories("${gtest_SOURCE_DIR}/include"
      "${gmock_SOURCE_DIR}/include"
      )
  endif()

  # Now simply link your own targets against gtest, gmock,
  # etc. as appropriate

    add_subdirectory(test)
endif()

Message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
