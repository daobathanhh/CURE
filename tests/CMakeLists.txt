# Tests CMakeLists.txt

# KD-tree tests
add_executable(test_kdtree test_kdtree.cpp)
target_link_libraries(test_kdtree PRIVATE cure_cpp)
add_test(NAME test_kdtree COMMAND test_kdtree)

# CURE algorithm tests
add_executable(test_cure test_cure.cpp)
target_link_libraries(test_cure PRIVATE cure_cpp)
add_test(NAME test_cure COMMAND test_cure)

# Distance function tests
add_executable(test_distance test_distance.cpp)
target_link_libraries(test_distance PRIVATE cure_cpp)
add_test(NAME test_distance COMMAND test_distance)

# Comparison tests (C++ vs Python)
add_executable(test_comparison comparison/test_comparison.cpp)
target_link_libraries(test_comparison PRIVATE cure_cpp)
add_test(NAME test_comparison COMMAND test_comparison 
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# C++ vs true labels only (Euclidean + Pearson, no Python)
add_executable(test_cpp_vs_true cpp_vs_true/test_cpp_vs_true.cpp)
target_link_libraries(test_cpp_vs_true PRIVATE cure_cpp)
add_test(NAME test_cpp_vs_true COMMAND test_cpp_vs_true
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
