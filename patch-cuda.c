#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <limits.h>

#include <unistd.h>
#define __USE_GNU
#include <dlfcn.h>

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
    size_t totalConstMem;
    int major;
    int minor;
    int clockRate;
    size_t textureAlignment;
    int deviceOverlap;
    int multiProcessorCount;
    int kernelExecTimeoutEnabled;
    int integrated;
    int canMapHostMemory;
    int computeMode;
    int concurrentKernels;
    int ECCEnabled;
    int pciBusID;
    int pciDeviceID;
    int tccDriver;
};

typedef int cudaError_t;
typedef void* cudaStream_t;
typedef cudaError_t (*orig_cudaGetDevProp_ftype)(struct cudaDeviceProp *, int);

static orig_cudaGetDevProp_ftype orig_getdevprop = NULL;

 #define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })


cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int deviceId) {
    char *endptr;
    const char *gpu_mem_limit_str = getenv("BACKEND_GPU_MEMORY_LIMIT");
    const char *gpu_proc_limit_str = getenv("BACKEND_GPU_PROCESSOR_LIMIT");
    size_t gpu_mem_limit = 0;
    int gpu_proc_limit = 0;

    if (gpu_mem_limit_str != NULL)
        gpu_mem_limit = strtoull(gpu_mem_limit_str, &endptr, 10);
    if (gpu_proc_limit_str != NULL)
        gpu_proc_limit = strtol(gpu_proc_limit_str, &endptr, 10);

    fprintf(stderr, "CUDA GET-DEV-PROPS HOOK: %lu, %d\n",
            gpu_mem_limit, gpu_proc_limit);
    if (orig_getdevprop == NULL) {
        orig_getdevprop = (orig_cudaGetDevProp_ftype)
                dlsym(RTLD_NEXT, "cudaGetDeviceProperties");
    }
    assert(orig_getdevprop != NULL);
    cudaError_t ret = orig_getdevprop(prop, deviceId);
    if (prop != NULL) {
        if (!(gpu_mem_limit == ULLONG_MAX || gpu_mem_limit == 0))
            prop->totalGlobalMem = min(gpu_mem_limit,
                                       prop->totalGlobalMem);
        if (!(gpu_proc_limit == LONG_MAX || gpu_proc_limit == 0))
            prop->multiProcessorCount = min(gpu_proc_limit,
                                            prop->multiProcessorCount);
    }
    return ret;
}
