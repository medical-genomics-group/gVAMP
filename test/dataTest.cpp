#include <gtest/gtest.h>
#include "data.hpp"

TEST(DataTester, readFamFileTest){
  Data data;
  std::string testFile {TEST_DATA};
  testFile +=  "uk10k_chr1_1mb.fam";
  data.readFamFile(testFile);
  ASSERT_EQ(data.numInds,3642);
}
