#define _GNU_SOURCE   // enable GNU extensions

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>

#include <unistd.h>
#include <dlfcn.h>


static void *orig_libc = NULL;
static void *orig_libcuda = NULL;
static void *orig_libcudart = NULL;


static inline void *_ensure_libc()
{
    if (orig_libc == NULL) {
        orig_libc = dlopen("libc.so", RTLD_LAZY);
    }
    assert(orig_libc != NULL);
    return orig_libc;
}


static inline void *_ensure_libcuda()
{
    if (orig_libcuda == NULL) {
        orig_libcuda = dlopen("libcuda.so", RTLD_LAZY);
    }
    assert(orig_libcuda != NULL);
    return orig_libcuda;
}


static inline void *_ensure_libcudart()
{
    if (orig_libcudart == NULL) {
        orig_libcudart = dlopen("libcudart.so", RTLD_LAZY);
    }
    assert(orig_libcudart != NULL);
    return orig_libcudart;
}


/* Below OVERRIDE_* macros are to write override functions
 * for specific library symbolsn with reference to the
 * original symbol. */

#define OVERRIDE_LIBC_SYMBOL(rettype, symbol, ...) \
typedef rettype (*orig_##symbol##_ftype)(__VA_ARGS__); \
static orig_##symbol##_ftype orig_##symbol = NULL; \
rettype symbol(__VA_ARGS__) { \
    do { \
        if (orig_##symbol == NULL) { \
            void *lib = _ensure_libc(); \
            orig_##symbol = (orig_##symbol##_ftype) \
                    dlsym(lib, #symbol); \
        } \
        assert(orig_##symbol != NULL); \
    } while (0);


#define OVERRIDE_CU_SYMBOL(rettype, symbol, ...) \
typedef rettype (*orig_##symbol##_ftype)(__VA_ARGS__); \
static orig_##symbol##_ftype orig_##symbol = NULL; \
rettype symbol(__VA_ARGS__) { \
    do { \
        if (orig_##symbol == NULL) { \
            void *lib = _ensure_libcuda(); \
            orig_##symbol = (orig_##symbol##_ftype) \
                    dlsym(lib, #symbol); \
        } \
        assert(orig_##symbol != NULL); \
    } while (0);


#define OVERRIDE_CUDA_SYMBOL(rettype, symbol, ...) \
typedef rettype (*orig_##symbol##_ftype)(__VA_ARGS__); \
static orig_##symbol##_ftype orig_##symbol = NULL; \
rettype symbol(__VA_ARGS__) { \
    do { \
        if (orig_##symbol == NULL) { \
            void *lib = _ensure_libcudart(); \
            orig_##symbol = (orig_##symbol##_ftype) \
                    dlsym(lib, #symbol); \
        } \
        assert(orig_##symbol != NULL); \
    } while (0);


#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

#define MAX_KLEN 256
#define MAX_VLEN 256
#define MAX_PATH 260


static inline int read_config
(const char *domain, const char *key, char *value, size_t len)
{
    bool found = false;
    char filename[MAX_PATH] = {0};
    snprintf(filename, MAX_PATH, ".config/%s.txt", domain);
    FILE *f = fopen(filename, "r");
    if (f != NULL) {
        char buffer[MAX_KLEN + MAX_VLEN];
        while(fgets(buffer, MAX_KLEN + MAX_VLEN, f)) {
            size_t l = strnlen(buffer, MAX_KLEN + MAX_VLEN);
            if (l > 0 && buffer[l - 1] == '\n')
                buffer[l - 1] = '\0';
            char *tok = &buffer[0];
            char *end = tok;
            char line_key[MAX_KLEN];
            // split only once with "="
            strsep(&end, "=");
            strncpy(line_key, tok, MAX_KLEN);
            tok = end;
            if (strncmp(line_key, key, MAX_KLEN) == 0) {
                strncpy(value, tok, MAX_VLEN);
                found = true;
                break;
            }
        }
        fclose(f);
    }
    if (!found) {
        char env_key[256] = {0};
        snprintf(env_key, 256, "BACKEND_%s", key);
        const char *env_value = getenv(env_key);
        if (env_value == NULL) {
            return -1;
        }
        snprintf(value, len, "%s", env_value);
    }
    return 0;
}
