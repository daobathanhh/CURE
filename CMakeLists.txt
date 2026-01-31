cmake_minimum_required(VERSION 3.14)

project(cure_cpp
    VERSION 1.0.0
    DESCRIPTION "CURE Clustering Algorithm C++ Implementation"
    LANGUAGES CXX
)

# Options
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(CURE_BUILD_EXAMPLES "Build examples" ON)
option(CURE_BUILD_TESTS "Build tests" ON)
option(CURE_BUILD_BENCHMARKS "Build benchmarks" OFF)

# Standard C++ settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

# Installation
include(GNUInstallDirs)

install(TARGETS cure_cpp
    EXPORT cure_cpp_targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

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
