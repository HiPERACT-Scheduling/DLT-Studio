/*
 * examples/c/hello_dls.c
 *
 * Solve a 3-processor instance via the C-ABI binding (libdls_c.so).
 * No C++ required — load the library at runtime with dlopen.
 *
 * Build:
 *   gcc hello_dls.c -ldl -o hello_dls
 *
 * Run:
 *   DLS_LIB=../../build/bin/libdls_c.so ./hello_dls
 */

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

typedef char* (*dls_solve_fn)(const char*, const char*, const char*);
typedef void  (*dls_free_fn)(char*);

/*
 * Instance text format (one header line + one line per processor):
 *   V <totalLoad>
 *   <S> <C> <A> <B> <p> <r> <d> <f> <l>
 *
 * B=1e18 = unbounded memory buffer; p/r/d/f/l = 0 (unused here).
 * Same instance as the C++ and Python examples for easy comparison.
 */
static const char INSTANCE[] =
    "V 1000.0\n"
    "0.0 0.1 0.5 1e+18 0.0 0.0 0.0 0.0 0.0\n"   /* P0 */
    "0.1 0.2 0.3 1e+18 0.0 0.0 0.0 0.0 0.0\n"   /* P1 */
    "0.2 0.3 0.2 1e+18 0.0 0.0 0.0 0.0 0.0\n";  /* P2 */

int main(void) {
    const char* path = getenv("DLS_LIB");
    if (!path) path = "../../build/bin/libdls_c.so";

    void* lib = dlopen(path, RTLD_LAZY);
    if (!lib) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }

    dls_solve_fn solve = (dls_solve_fn)dlsym(lib, "dls_solve");
    dls_free_fn  free_ = (dls_free_fn) dlsym(lib, "dls_free");
    if (!solve || !free_) { fprintf(stderr, "dlsym failed\n"); return 1; }

    /* solver = "auto", no extra options */
    char* result = solve(INSTANCE, "auto", "");
    printf("%s\n", result);
    free_(result);

    dlclose(lib);
    return 0;
}
