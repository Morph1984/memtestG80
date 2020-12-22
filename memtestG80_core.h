/*
 * memtestG80_core.h
 * Public API for core memory test functions for MemtestG80
 * Includes functional and OO interfaces to GPU test functions.
 *
 * Author: Imran Haque, 2009
 * Copyright 2009, Stanford University
 *
 * This file is licensed under the terms of the LGPL. Please see
 * the COPYING file in the accompanying source distribution for
 * full license terms.
 *
 */
#ifndef _MEMTESTG80_CORE_H_
#define _MEMTESTG80_CORE_H_

#include "types.h"

#if defined(WINDOWS) || defined(WINNV)
#include <windows.h>
inline u32 getTimeMilliseconds(void) {
    return GetTickCount();
}
#include <windows.h>
#define SLEEPMS(x) Sleep(x)
#elif defined(LINUX) || defined(OSX)
#include <sys/time.h>
inline u32 getTimeMilliseconds(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}
#include <unistd.h>
#define SLEEPMS(x) usleep(x * 1000)
#else
#error Must #define LINUX, WINDOWS, WINNV, or OSX
#endif

// By default the driver will spinwait when blocked on a kernel call
// Use the SOFTWAIT macro to replace this with a thread sleep and occasional poll
// limit expresses the max time we're willing to stay in the sleep loop - default = 15sec
inline s32 _pollStatus(unsigned length = 1, unsigned limit = 15000) {
    //while (cudaStreamQuery(0) != cudaSuccess) {SLEEPMS(length);}
    unsigned startTime = getTimeMilliseconds();
    while (cudaStreamQuery(0) == cudaErrorNotReady) {
        if ((getTimeMilliseconds() - startTime) > limit)
            return -1;
        SLEEPMS(length);
    }
    return 0;
}
#define SOFTWAIT()                                                             \
    if (_pollStatus() != 0) {                                                  \
        return 0xFFFFFFFE;                                                     \
    } // -2
#define SOFTWAIT_LIM(lim)                                                      \
    if (_pollStatus(1, lim) != 0) {                                            \
        return 0xFFFFFFFE;                                                     \
    } // -2
//#define SOFTWAIT()
//#define SOFTWAIT(delay) if (_pollStatus(delay) != 0) return -2;
//#define SOFTWAIT(delay,limit) if (_pollStatus(delay,limit) != 0) return -2;
//#define SOFTWAIT() while (cudaStreamQuery(0) != cudaSuccess) {SLEEPMS(1);}
//#define SOFTWAIT(x) while (cudaStreamQuery(0) != cudaSuccess) {SLEEPMS(x);}
//#define SOFTWAIT()

// Use this macro to check for kernel errors
#define CHECK_LAUNCH_ERROR()                                                   \
    if (cudaGetLastError() != cudaSuccess) {                                   \
        return 0xFFFFFFFF; /* -1 */                                            \
    }

// OO interface to MemtestG80 functions
class memtestState {
protected:
    const u32 nBlocks;
    const u32 nThreads;
    u32 loopIters;
    u32 megsToTest;
    s32 lcgPeriod;
    u32* devTestMem;
    u32* devTempMem;
    u32* hostTempMem;
    bool allocated;

public:
    u32 initTime;
    memtestState()
        : nBlocks(1024), nThreads(512), allocated(false), devTestMem(NULL),
          devTempMem(NULL), hostTempMem(NULL), initTime(0), lcgPeriod(1024){};
    ~memtestState() {
        deallocate();
    }

    u32 allocate(u32 mbToTest);
    void deallocate();
    bool isAllocated() const {
        return allocated;
    }
    u32 size() const {
        return megsToTest;
    }
    void setLCGPeriod(s32 period) {
        lcgPeriod = period;
    }
    s32 getLCGPeriod() const {
        return lcgPeriod;
    }

    bool gpuMemoryBandwidth(double& bandwidth, u32 mbToTest, u32 iters = 5);
    bool gpuWriteConstant(const u32 constant) const;
    bool gpuVerifyConstant(u32& errorCount, const u32 constant) const;
    bool gpuShortLCG0(u32& errorCount, const u32 repeats) const;
    bool gpuShortLCG0Shmem(u32& errorCount, const u32 repeats) const;
    bool gpuMovingInversionsOnesZeros(u32& errorCount) const;
    bool gpuWalking8BitM86(u32& errorCount, const u32 shift) const;
    bool gpuWalking8Bit(u32& errorCount, const bool ones,
                        const u32 shift) const;
    bool gpuMovingInversionsRandom(u32& errorCount) const;
    bool gpuWalking32Bit(u32& errorCount, const bool ones,
                         const u32 shift) const;
    bool gpuRandomBlocks(u32& errorCount, const u32 seed) const;
    bool gpuModuloX(u32& errorCount, const u32 shift, const u32 pattern,
                    const u32 modulus, const u32 overwriteIters) const;
};

// Utility functions
__host__ double gpuMemoryBandwidth(u32* src, u32* dst, u32 mbToTest, u32 iters);
__host__ void gpuWriteConstant(const u32 nBlocks, const u32 nThreads, u32* base,
                               u32 N, const u32 constant);
__host__ u32 gpuVerifyConstant(const u32 nBlocks, const u32 nThreads, u32* base,
                               u32 N, const u32 constant, u32* blockErrorCount,
                               u32* errorCounts);

__host__ void cpuWriteConstant(const u32 nBlocks, const u32 nThreads, u32* base,
                               u32 N, const u32 constant);
__host__ u32 cpuVerifyConstant(const u32 nBlocks, const u32 nThreads, u32* base,
                               u32 N, const u32 constant);

// Logic tests
__host__ u32 gpuShortLCG0(const u32 nBlocks, const u32 nThreads, u32* base,
                          u32 N, const u32 repeats, const s32 period,
                          u32* blockErrorCounts, u32* errorCounts);
__host__ u32 gpuShortLCG0Shmem(const u32 nBlocks, const u32 nThreads, u32* base,
                               u32 N, const u32 repeats, const s32 period,
                               u32* blockErrorCounts, u32* errorCounts);

// Memtest86 Test 2: tseq=0,4
__host__ u32 gpuMovingInversionsOnesZeros(const u32 nBlocks, const u32 nThreads,
                                          u32* base, u32 N,
                                          u32* blockErrorCounts,
                                          u32* errorCounts);

// Memtest86 Test 3: tseq=1
__host__ u32 gpuWalking8BitM86(const u32 nBlocks, const u32 nThreads, u32* base,
                               u32 N, u32 shift, u32* blockErrorCounts,
                               u32* errorCounts);
__host__ u32 cpuWalking8BitM86(const u32 nBlocks, const u32 nThreads, u32* base,
                               u32 N, u32 shift);
__host__ u32 gpuWalking8Bit(const u32 nBlocks, const u32 nThreads, u32* base,
                            u32 N, bool ones, u32 shift, u32* blockErrorCount,
                            u32* errorCounts);

// Memtest86 Test 4: tseq=10
__host__ u32 gpuMovingInversionsRandom(const u32 nBlocks, const u32 nThreads,
                                       u32* base, u32 N, u32* blockErrorCounts,
                                       u32* errorCounts);

// Memtest86 Test 6: tseq=2
__host__ u32 gpuWalking32Bit(const u32 nBlocks, const u32 nThreads, u32* base,
                             u32 N, bool ones, u32 shift, u32* blockErrorCount,
                             u32* errorCounts);
//
// Memtest86 Test 7: tseq=9
__host__ u32 gpuRandomBlocks(const u32 nBlocks, const u32 nThreads, u32* base,
                             u32 N, u32 seed, u32* blockErrorCount,
                             u32* errorCounts);

// Memtest86 Test 8: tseq=3 (M86 uses modulus = 20)
__host__ u32 gpuModuloX(const u32 nBlocks, const u32 nThreads, u32* base,
                        const u32 N, u32 shift, u32 pattern1, const u32 modulus,
                        const u32 iters, u32* blockErrorCount,
                        u32* errorCounts);

#endif
