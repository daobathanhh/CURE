# 3.7+ for build-in-container (Debian Stretch has 3.7.2); 3.14+ preferred on host
cmake_minimum_required(VERSION 3.7)

project(cure_cpp
    VERSION 1.0.0
    LANGUAGES CXX
)

# Options
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(CURE_BUILD_EXAMPLES "Build examples" ON)
option(CURE_BUILD_TESTS "Build tests" ON)
option(CURE_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(CURE_BUILD_DISTRIBUTED_KMEANS "Build distributed K-means (for comparison with CURE)" OFF)

# Standard C++ settings
# CMake 3.7 does not know CXX17; avoid setting STANDARD 17 so no target "requires" it, and pass -std=c++17 by hand
if(CMAKE_VERSION VERSION_LESS "3.8")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
else()
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

# Optimization flags
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# Add source directory (builds the library)
add_subdirectory(src)

# Examples
if(CURE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Tests
if(CURE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Benchmarks
if(CURE_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Distributed K-means (standalone; for comparison with CURE, does not touch cure_cpp)
if(CURE_BUILD_DISTRIBUTED_KMEANS)
    add_subdirectory(distributed_kmeans)
endif()

# Installation (install TARGETS must be in same dir as target for CMake 3.7; see src/CMakeLists.txt)
include(GNUInstallDirs)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT cure_cpp_targets
    FILE cure_cpp-config.cmake
    NAMESPACE cure::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cure_cpp
)

# Package configuration
include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/cure_cpp-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/cure_cpp-config-version.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cure_cpp
)
