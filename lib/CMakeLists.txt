# ===========================================================================
# DLS library (the backend) — the self-sustained C++ library + CLIs + tests.
#
# This subproject knows NOTHING about the binding or the GUI; it builds and is
# usable entirely on its own:
#     cmake -B build-lib lib && cmake --build build-lib   # library + CLIs + tests
# Optional HiGHS LP/MILP backend via vcpkg (-DDLS_WITH_HIGHS=ON). The root
# umbrella CMakeLists adds this plus the optional binding; nothing here depends
# outward (see the separate-backend-frontend rule).
# ===========================================================================

cmake_minimum_required(VERSION 3.16)

# Auto-select the in-repo vcpkg toolchain when HiGHS is requested (must run
# before project()). Harmless/unused for the dependency-free build.
if(DLS_WITH_HIGHS AND NOT CMAKE_TOOLCHAIN_FILE
   AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake")
  set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_LIST_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake"
      CACHE STRING "Vcpkg toolchain (in-repo)")
  set(_dls_bincache "${CMAKE_CURRENT_LIST_DIR}/vcpkg/.binarycache")
  file(MAKE_DIRECTORY "${_dls_bincache}")
  set(ENV{VCPKG_DEFAULT_BINARY_CACHE} "${_dls_bincache}")
endif()

project(dls_lib CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()
add_compile_options(-Wall -Wextra)

# Header root so includes resolve uniformly as "core/...", "util/...", etc.
# (this directory, lib/). Exposed to anything that links the library
# (e.g. the binding) via the dls_includes INTERFACE target below.
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_library(dls_includes INTERFACE)
target_include_directories(dls_includes INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# All artifacts land in <top-build>/bin regardless of which dir configured.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(Threads REQUIRED)

# Shared timing helper compiled once and reused by the CLIs (and the binding).
add_library(dls_util STATIC util/time.cpp)
target_link_libraries(dls_util PUBLIC dls_includes Threads::Threads)

# ---- solvers / CLIs -------------------------------------------------------
add_executable(ga heuristics/ga/ga.cpp)
target_link_libraries(ga PRIVATE dls_util)

add_executable(dls cli/dls_main.cpp)
target_link_libraries(dls PRIVATE dls_util)

add_executable(dls-bench bench/bench_main.cpp)
target_link_libraries(dls-bench PRIVATE dls_util)

add_executable(hello_dls examples/hello_dls.cpp)
target_link_libraries(hello_dls PRIVATE dls_util)

# ---- optional HiGHS LP/MILP backend ---------------------------------------
option(DLS_WITH_HIGHS "Build the optional HiGHS LP backend" OFF)
if(DLS_WITH_HIGHS)
  find_package(HIGHS CONFIG REQUIRED)
  add_library(dls_highs STATIC
    engines/highs/highs_schedule_evaluator.cpp
    engines/highs/highs_lp_solver.cpp
    exact/milp/milp_solver.cpp
    exact/milp/multi_milp_solver.cpp
    exact/milp/mlsd_milp_solver.cpp)
  target_link_libraries(dls_highs PUBLIC highs::highs dls_includes)
  target_compile_definitions(dls_highs PUBLIC DLS_WITH_HIGHS)
  target_link_libraries(ga PRIVATE dls_highs)
  target_link_libraries(dls PRIVATE dls_highs)
  target_link_libraries(dls-bench PRIVATE dls_highs)

  add_executable(portfolio_compare examples/portfolio_compare.cpp)
  target_link_libraries(portfolio_compare PRIVATE dls_highs dls_util)
endif()

# ---- tests (doctest, single translation unit) -----------------------------
enable_testing()
add_executable(dls_tests tests/tests.cpp)
target_include_directories(dls_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(dls_tests PRIVATE dls_util)
if(DLS_WITH_HIGHS)
  target_link_libraries(dls_tests PRIVATE dls_highs)
endif()
add_test(NAME dls_tests COMMAND dls_tests)
