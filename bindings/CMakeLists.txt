# ===========================================================================
# C-ABI binding (adapter) — libdls_c.so.
#
# A thin, optional adapter over the backend library for front-ends (the Python
# package in frontend/dls loads it via ctypes). It depends ONLY inward on the
# backend's targets (dls_util for the headers/timing, dls_highs when present);
# the library never depends on it. Built from the umbrella when
# -DDLS_WITH_BINDINGS=ON (default). Requires the backend subproject to have been
# added first (the umbrella does so).
# ===========================================================================

add_library(dls_c SHARED dls_c.cpp)
# Land alongside the CLIs in <top-build>/bin so front-ends find it predictably.
set_target_properties(dls_c PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
target_link_libraries(dls_c PRIVATE dls_util)        # headers + timing + threads
if(DLS_WITH_HIGHS)
  target_link_libraries(dls_c PRIVATE dls_highs)     # GUI sees the MILP solvers too
endif()
