# Source CMakeLists.txt

# Library source files
set(CURE_SOURCES
    cure/distance.cpp
    cure/kd_tree.cpp
    cure/cluster.cpp
    cure/cure.cpp
)

# Create the library
add_library(cure_cpp ${CURE_SOURCES})

# Set include directories
target_include_directories(cure_cpp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Set compile features
target_compile_features(cure_cpp PUBLIC cxx_std_17)

# Export symbols for shared library (default visibility for simplicity)
set_target_properties(cure_cpp PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Add alias for modern CMake usage
add_library(cure::cure ALIAS cure_cpp)
