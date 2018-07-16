#include "patch.h"
#include "cuda-interop.h"

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <limits.h>
#include <dirent.h>


static void *orig_libcuda = NULL;
static void *orig_libcudart = NULL;


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
        char filename[MAX_PATH] = {0};
        bool found = false;
        DIR *d = NULL;
        d = opendir("/usr/local/cuda/lib64");
        if (d) {
            struct dirent *dir = NULL;
            while ((dir = readdir(d)) != NULL) {
                if (has_prefix("libcudart.so.", dir->d_name)) {
                    strncpy(filename, dir->d_name, MAX_PATH);
                    found = true;
                    break;
                }
            }
            closedir(d);
        }
        if (found) {
            orig_libcudart = dlopen(filename, RTLD_LAZY);
        } else {
            // Try fallback with non-versioned filename
            orig_libcudart = dlopen("libcudart.so", RTLD_LAZY);
        }
    }
    assert(orig_libcudart != NULL);
    return orig_libcudart;
}


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


static inline size_t _get_configured_gpu_mem_limit()
{
    char *endptr = NULL;
    char gpu_mem_limit_str[MAX_VLEN];
    int ret = read_config("gpu", "GPU_MEMORY_LIMIT",
                          gpu_mem_limit_str, MAX_VLEN);
    size_t gpu_mem_limit = 0;
    if (ret == 0)
        gpu_mem_limit = strtoull(gpu_mem_limit_str, &endptr, 10);
    if (gpu_mem_limit == ULLONG_MAX)
        gpu_mem_limit = 0;
    return gpu_mem_limit;
}


static int _get_configured_gpu_proc_limit()
{
    char *endptr = NULL;
    char gpu_proc_limit_str[MAX_VLEN];
    int ret = read_config("gpu", "GPU_PROCESSOR_LIMIT",
                          gpu_proc_limit_str, MAX_VLEN);
    int gpu_proc_limit = 0;
    if (ret == 0)
        gpu_proc_limit = strtol(gpu_proc_limit_str, &endptr, 10);
    if (gpu_proc_limit == INT_MAX)
        gpu_proc_limit = 0;
    return gpu_proc_limit;
}


/* Override driver APIs */

OVERRIDE_CU_SYMBOL(int, cuDeviceGetName,
                   char* name, int len, CUdevice dev)
    char orig_name[256];
    int ret = orig_cuDeviceGetName(orig_name, 256, dev);
    if (ret == CUDA_SUCCESS) {
        // Hide original GPU name.
        memset(name, 0, len);
        snprintf(name, len, "CUDA GPU");
    }
    return ret;
}


OVERRIDE_CU_SYMBOL(int, cuMemGetInfo_v2,
                   size_t *free, size_t *total)
    int ret = orig_cuMemGetInfo_v2(free, total);
    size_t limit = _get_configured_gpu_mem_limit();
    if (ret == CUDA_SUCCESS && limit != 0) {
        *free = min(limit, *free);
        *total = min(limit, *total);
    }
    return ret;
}


int cuMemGetInfo(size_t *free, size_t *total)
{
    return cuMemGetInfo_v2(free, total);
}


OVERRIDE_CU_SYMBOL(int, cuDeviceTotalMem_v2,
                   size_t *bytes, CUdevice dev)
    int ret = orig_cuDeviceTotalMem_v2(bytes, dev);
    size_t limit = _get_configured_gpu_mem_limit();
    if (ret == CUDA_SUCCESS && limit != 0) {
        *bytes = min(limit, *bytes);
    }
    return ret;
}

int cuDeviceTotalMem(size_t *bytes, CUdevice dev)
{
    return cuDeviceTotalMem_v2(bytes, dev);
}


OVERRIDE_CU_SYMBOL(int, cuDeviceGetAttribute,
                   int *pi, int attrib, CUdevice dev)
    int ret = orig_cuDeviceGetAttribute(pi, attrib, dev);
    int proc_limit = 0;
    if (ret == CUDA_SUCCESS) {
        switch (attrib) {
        case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT:
            proc_limit = _get_configured_gpu_proc_limit();
            if (proc_limit != 0) {
                *pi = min(proc_limit, *pi);
            }
            break;
        default:
            break;
        }
    }
    return ret;
}


/* Override runtime APIs */

int cudaMemGetInfo(size_t *free, size_t *total)
{
    return cuMemGetInfo(free, total);
}


OVERRIDE_CUDA_SYMBOL(cudaError_t, cudaGetDeviceProperties,
                     struct cudaDeviceProp *prop, int deviceId)
    size_t gpu_mem_limit = _get_configured_gpu_mem_limit();
    int gpu_proc_limit = _get_configured_gpu_proc_limit();
    cudaError_t ret = orig_cudaGetDeviceProperties(prop, deviceId);
    if (ret == CUDA_SUCCESS && prop != NULL) {
        memset(prop->name, 0, 256);
        snprintf(prop->name, 256, "CUDA GPU");
        if (gpu_mem_limit != 0)
            prop->totalGlobalMem = min(gpu_mem_limit,
                                       prop->totalGlobalMem);
        if (gpu_proc_limit != 0)
            prop->multiProcessorCount = min(gpu_proc_limit,
                                            prop->multiProcessorCount);
    }
    return ret;
}
