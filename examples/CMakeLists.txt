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

# CLI: run CURE on CSV, write labels JSON (for RFM plotting)
add_executable(run_cure_csv run_cure_csv.cpp)
target_link_libraries(run_cure_csv PRIVATE cure_cpp)

# Batch: run normal + scalable CURE (Euclidean + Pearson) for RFM sweep; Python only plots
add_executable(run_rfm_batch run_rfm_batch.cpp)
target_link_libraries(run_rfm_batch PRIVATE cure_cpp)
