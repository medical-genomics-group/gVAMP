set(BINARY ${CMAKE_PROJECT_NAME}_test)

add_executable(${BINARY}
  dataTest.cpp
  main.cpp
  )


set_property(TARGET ${BINARY} PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET ${BINARY} PROPERTY CXX_STANDARD 17)
target_compile_definitions(${BINARY} PRIVATE TEST_DATA="${CMAKE_CURRENT_SOURCE_DIR}/../test/data/")

target_link_libraries(${BINARY}  ${CMAKE_PROJECT_NAME}_lib gtest MPI::MPI_CXX  OpenMP::OpenMP_CXX)


add_test(NAME ${BINARY} COMMAND ${BINARY})
