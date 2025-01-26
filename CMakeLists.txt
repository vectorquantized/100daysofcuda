cmake_minimum_required(VERSION 3.10)
project(cudakernels CUDA CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# using fmt for printing
find_package(fmt REQUIRED)
# Include directories for headers
include_directories(${PROJECT_SOURCE_DIR}/src)

# Collect common CPP and CPU sources
file(GLOB CPP_SOURCES ${PROJECT_SOURCE_DIR}/src/csrc/*.cpp)
file(GLOB CPU_SOURCES ${PROJECT_SOURCE_DIR}/src/cpu/*.cpp)

# Collect all day_* directories dynamically
file(GLOB DAY_DIRS ${PROJECT_SOURCE_DIR}/src/day_*)

# Iterate over day_* directories to create individual targets
foreach(DAY_DIR ${DAY_DIRS})
    # Extract the directory name (e.g., day_1)
    get_filename_component(DAY_NAME ${DAY_DIR} NAME)

    # Collect all .cu files in the current day_* directory
    file(GLOB CUDA_FILES ${DAY_DIR}/*.cu)

    # Create a target for the current day directory
    add_executable(${DAY_NAME} ${CUDA_FILES} ${CPP_SOURCES} ${CPU_SOURCES})

    # Set CUDA architectures
    set_target_properties(${DAY_NAME} PROPERTIES CUDA_ARCHITECTURES "80;86;89;90")

    # Add include directories
    target_include_directories(${DAY_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/src)

    # Link fmt and 
    target_link_libraries(${DAY_NAME} PRIVATE fmt::fmt)

endforeach()

# Custom clean target for full rebuild
add_custom_target(clean_build
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_BINARY_DIR}/*
    COMMENT "Performing a full clean of the build directory"
)
