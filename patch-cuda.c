#define _GNU_SOURCE   // enable GNU extensions

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include <limits.h>

#include <unistd.h>
#include <dlfcn.h>

#define MAX_KLEN 256
#define MAX_VLEN 256

/* Taken from NVIDIA's official documentation (CUDA 9.0) */
struct cudaDeviceProp {
    char name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    size_t memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    size_t totalConstMem;
    int major;
    int minor;
    size_t textureAlignment;
    size_t texturePitchAlignment;
    int deviceOverlap;
    int multiProcessorCount;
    int kernelExecTimeoutEnabled;
    int integrated;
    int canMapHostMemory;
    int computeMode;
    int maxTexture1D;
    int maxTexture1DMipmap;
    int maxTexture1DLinear;
    int maxTexture2D[2];
    int maxTexture2DMipmap[2];
    int maxTexture2DLinear[3];
    int maxTexture2DGather[2];
    int maxTexture3D[3];
    int maxTexture3DAlt[3];
    int maxTextureCubemap;
    int maxTexture1DLayered[2];
    int maxTexture2DLayered[3];
    int maxTextureCubemapLayered[2];
    int maxSurface1D;
    int maxSurface2D[2];
    int maxSurface3D[3];
    int maxSurface1DLayered[2];
    int maxSurface2DLayered[3];
    int maxSurfaceCubemap;
    int maxSurfaceCubemapLayered[2];
    size_t surfaceAlignment;
    int concurrentKernels;
    int ECCEnabled;
    int pciBusID;
    int pciDeviceID;
    int tccDriver;
    int asyncEngineCount;
    int unifiedAddressing;
    int memoryClockRate;
    int memoryBusWidth;
    int l2CacheSize;
    int maxThreadsPerMultiProcessor;
    int streamPrioritiesSupported;
    int globalL1CacheSupported;
    int localL1CacheSupported;
    size_t sharedMemPerMultiprocessor;
    int regsPerMultiprocessor;
    int managedMemSupported;
    int isMultiGpuBoard;
    int multiGpuBoardGroupID;
    int pageableMemoryAccess;
    int concurrentManagedAccess;
    int computePreemptionSupported;
    int canUseHostPointerForRegisteredMem;
    int cooperativeLaunch;
    int cooperativeMultiDeviceLaunch;
    int pageableMemoryAccessUsesHostPageTables;
    int directManagedMemAccessFromHost;
    char __future_buffer[4096];
};

#define CU_DEVICE_CPU ((CUdevice)-1)
#define CU_DEVICE_INVALID ((CUdevice)-2)
#define CU_IPC_HANDLE_SIZE 64
#define CUDA_SUCCESS 0

#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT 16

typedef int CUresult;
typedef int CUdevice;
typedef unsigned int CUdeviceptr;

#define REPLACE_CU_SYMBOL(symbol, ...) \
typedef int (*orig_##symbol##_ftype)(__VA_ARGS__); \
static orig_##symbol##_ftype orig_##symbol = NULL; \
int symbol(__VA_ARGS__) { \
    do { \
        if (orig_##symbol == NULL) { \
            void *lib = _ensure_libcuda(); \
            orig_##symbol = (orig_##symbol##_ftype) \
                    dlsym(lib, #symbol); \
        } \
        assert(orig_##symbol != NULL); \
    } while (0);

#define REPLACE_CUDA_SYMBOL(symbol, ...) \
typedef int (*orig_##symbol##_ftype)(__VA_ARGS__); \
static orig_##symbol##_ftype orig_##symbol = NULL; \
int symbol(__VA_ARGS__) { \
    do { \
        if (orig_##symbol == NULL) { \
            void *lib = _ensure_libcudart(); \
            orig_##symbol = (orig_##symbol##_ftype) \
                    dlsym(lib, #symbol); \
        } \
        assert(orig_##symbol != NULL); \
    } while (0);

typedef int cudaError_t;
typedef void* cudaStream_t;

static void *orig_libcuda = NULL;
static void *orig_libcudart = NULL;

 #define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })


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


static inline int read_config(const char *key, char *value, size_t len)
{
    bool found = false;
    FILE *f = fopen(".config/gpu.txt", "r");
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


static inline size_t _get_configured_gpu_mem_limit()
{
    char *endptr = NULL;
    char gpu_mem_limit_str[MAX_VLEN];
    int ret = read_config("GPU_MEMORY_LIMIT", gpu_mem_limit_str, MAX_VLEN);
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
    int ret = read_config("GPU_PROCESSOR_LIMIT", gpu_proc_limit_str, MAX_VLEN);
    int gpu_proc_limit = 0;
    if (ret == 0)
        gpu_proc_limit = strtol(gpu_proc_limit_str, &endptr, 10);
    if (gpu_proc_limit == INT_MAX)
        gpu_proc_limit = 0;
    return gpu_proc_limit;
}


/* Override driver APIs */

REPLACE_CU_SYMBOL(cuDeviceGetName, char* name, int len, CUdevice dev)
    char orig_name[256];
    int ret = orig_cuDeviceGetName(orig_name, 256, dev);
    if (ret == CUDA_SUCCESS) {
        // Hide original GPU name.
        memset(name, 0, len);
        snprintf(name, len, "CUDA GPU");
    }
    return ret;
}

REPLACE_CU_SYMBOL(cuMemGetInfo_v2, size_t *free, size_t *total)
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


REPLACE_CU_SYMBOL(cuDeviceTotalMem_v2, size_t *bytes, CUdevice dev)
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


REPLACE_CU_SYMBOL(cuDeviceGetAttribute, int *pi, int attrib, CUdevice dev)
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


REPLACE_CUDA_SYMBOL(cudaGetDeviceProperties, struct cudaDeviceProp *prop, int deviceId)
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
