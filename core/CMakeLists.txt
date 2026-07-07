cmake_minimum_required(VERSION 3.22)
project(bestefar_core CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

add_library(bestefar_core STATIC
    src/outer_circle.cpp
    src/screen.cpp
    src/rings.cpp
    src/perspective.cpp
    src/circles.cpp
    src/hits.cpp
    src/overlap.cpp
    src/scoring.cpp
    src/analyze.cpp
    src/autocapture.cpp
    src/ffi.cpp
)
target_include_directories(bestefar_core
    PUBLIC include
    PRIVATE src
)
target_link_libraries(bestefar_core PUBLIC ${OpenCV_LIBS})
target_include_directories(bestefar_core PUBLIC ${OpenCV_INCLUDE_DIRS})

if(NOT ANDROID AND NOT IOS)
    add_executable(bestefar_cli cli/main.cpp)
    target_link_libraries(bestefar_cli PRIVATE bestefar_core)
endif()
