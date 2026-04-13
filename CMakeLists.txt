cmake_minimum_required(VERSION 3.16)
project(autonomous_intent_estimator_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(intent_demo
    src/main.cpp
    src/intent_estimator.cpp
    src/csv_io.cpp
)

target_include_directories(intent_demo PRIVATE include)

if (MSVC)
    target_compile_options(intent_demo PRIVATE /W4)
else()
    target_compile_options(intent_demo PRIVATE -Wall -Wextra -Wpedantic)
endif()
