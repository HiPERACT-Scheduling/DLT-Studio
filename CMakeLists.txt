# ===========================================================================
# DLS — umbrella build wiring the separate subprojects together.
#
# Layout (one-way dependency: frontend -> binding -> library):
#   lib/   the self-sustained C++ library + CLIs + tests   (builds alone)
#   bindings/  optional C-ABI adapter (libdls_c.so) for front-ends
#   frontend/  the Streamlit GUI (Python; not built by CMake)
#   deploy/    ops (backup script, systemd unit)
#
# Common builds:
#   cmake -B build                                  # library + binding (dep-free)
#   cmake -B build-highs -DDLS_WITH_HIGHS=ON        # + HiGHS LP/MILP solvers
#   cmake -B build-lib lib                       # the LIBRARY ALONE (no binding)
# ===========================================================================

cmake_minimum_required(VERSION 3.16)

# Forwarded to the backend; declared here so it is visible before project().
option(DLS_WITH_HIGHS "Build the optional HiGHS LP backend" OFF)
option(DLS_WITH_BINDINGS "Build the C-ABI binding (libdls_c) for front-ends" ON)

# vcpkg lives under lib/; select its toolchain (and point the manifest there)
# before project() when HiGHS is requested, so the umbrella build resolves HiGHS.
if(DLS_WITH_HIGHS AND NOT CMAKE_TOOLCHAIN_FILE
   AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/lib/vcpkg/scripts/buildsystems/vcpkg.cmake")
  set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_LIST_DIR}/lib/vcpkg/scripts/buildsystems/vcpkg.cmake"
      CACHE STRING "Vcpkg toolchain (in-repo)")
  set(VCPKG_MANIFEST_DIR "${CMAKE_CURRENT_LIST_DIR}/lib" CACHE STRING "vcpkg.json location")
  set(_dls_bincache "${CMAKE_CURRENT_LIST_DIR}/lib/vcpkg/.binarycache")
  file(MAKE_DIRECTORY "${_dls_bincache}")
  set(ENV{VCPKG_DEFAULT_BINARY_CACHE} "${_dls_bincache}")
endif()

project(dls CXX)

add_subdirectory(lib)
if(DLS_WITH_BINDINGS)
  add_subdirectory(bindings)
endif()
