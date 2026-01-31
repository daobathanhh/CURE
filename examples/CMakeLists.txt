# Examples CMakeLists.txt

# Basic CURE example
add_executable(cure_example main.cpp)
target_link_libraries(cure_example PRIVATE cure_cpp)

# Benchmark example
add_executable(cure_benchmark benchmark.cpp)
target_link_libraries(cure_benchmark PRIVATE cure_cpp)

# KD-tree example
add_executable(kdtree_example kdtree_example.cpp)
target_link_libraries(kdtree_example PRIVATE cure_cpp)
