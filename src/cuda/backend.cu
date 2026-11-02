#include <algorithm>
#include <cstring>
#include <sstream>
#include <type_traits>

#include <cuda_runtime.h>

#if defined(KAEL_USE_CUBLAS)
#include <cublas_v2.h>
#endif

#include "kael/device_kernels.h"
#include "kael/error.h"

#include "internal/element_functions.h"

#define KAEL_CUDA_CHECK(call)                                                                      \
    do {                                                                                           \
        const cudaError_t kael_cuda_status_ = (call);                                              \
        if (kael_cuda_status_ != cudaSuccess) {                                                    \
            ::kael::device::fail_call(#call, __FILE__, __LINE__, kael_cuda_status_);               \
        }                                                                                          \
    } while (0)

namespace kael::device {

[[noreturn]] void fail_call(const char* call, const char* file, int line, cudaError_t status) {
    std::ostringstream os;
    os << "CUDA error at " << file << ":" << line << ": " << call << " failed with "
       << cudaGetErrorString(status);
    throw Error(os.str());
}

inline constexpr int kThreadsPerBlock = 256;
inline constexpr std::int64_t kMaxBlocks = 65535;

template <typename Task> __global__ void run_task(std::int64_t count, Task task) {
    std::int64_t element = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (; element < count; element += stride) {
        task(element);
    }
}

template <typename Task> void for_each(std::int64_t count, const Task& task) {
    if (count == 0) {
        return;
    }
    const auto blocks =
        static_cast<int>(std::min((count + kThreadsPerBlock - 1) / kThreadsPerBlock, kMaxBlocks));
    run_task<Task><<<blocks, kThreadsPerBlock>>>(count, task);
    KAEL_CUDA_CHECK(cudaGetLastError());
}

} // namespace kael::device

namespace kael::device::tasks {

template <typename T> __device__ void atomic_add(T* target, T value) {
    atomicAdd(target, value);
}

template <> __device__ void atomic_add<std::int64_t>(std::int64_t* target, std::int64_t value) {
    atomicAdd(reinterpret_cast<unsigned long long*>(target), static_cast<unsigned long long>(value));
}

} // namespace kael::device::tasks

#include "kernels.inl"

namespace kael::device {

bool available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

void synchronize() {
    KAEL_CUDA_CHECK(cudaDeviceSynchronize());
}

std::byte* allocate(std::int64_t bytes) {
    void* block = nullptr;
    KAEL_CUDA_CHECK(cudaMalloc(&block, static_cast<std::size_t>(bytes)));
    return static_cast<std::byte*>(block);
}

void release(std::byte* block) noexcept {
    cudaFree(block);
}

void copy_to_device(std::byte* target, const std::byte* source, std::int64_t bytes) {
    KAEL_CUDA_CHECK(cudaMemcpy(target, source, static_cast<std::size_t>(bytes),
                               cudaMemcpyHostToDevice));
}

void copy_to_host(std::byte* target, const std::byte* source, std::int64_t bytes) {
    KAEL_CUDA_CHECK(cudaMemcpy(target, source, static_cast<std::size_t>(bytes),
                               cudaMemcpyDeviceToHost));
}

void copy_within_device(std::byte* target, const std::byte* source, std::int64_t bytes) {
    KAEL_CUDA_CHECK(cudaMemcpy(target, source, static_cast<std::size_t>(bytes),
                               cudaMemcpyDeviceToDevice));
}

#if defined(KAEL_USE_CUBLAS)
namespace {

cublasHandle_t cublas_handle() {
    static cublasHandle_t handle = [] {
        cublasHandle_t created = nullptr;
        if (cublasCreate(&created) != CUBLAS_STATUS_SUCCESS) {
            throw Error("cublasCreate failed");
        }
        return created;
    }();
    return handle;
}

// cuBLAS is column-major: for row-major matrices it computes the transpose, Cᵀ = Bᵀ Aᵀ.
template <typename T, typename Gemm>
void cublas_multiply(Gemm gemm, const T* a, const T* b, T* out, const MatmulSizes& sizes) {
    const T one{1};
    const T zero{0};
    for (std::int64_t batch = 0; batch < sizes.batches; ++batch) {
        const cublasStatus_t status = gemm(
            cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(sizes.columns),
            static_cast<int>(sizes.rows), static_cast<int>(sizes.inner), &one,
            b + batch * sizes.inner * sizes.columns, static_cast<int>(sizes.columns),
            a + batch * sizes.rows * sizes.inner, static_cast<int>(sizes.inner), &zero,
            out + batch * sizes.rows * sizes.columns, static_cast<int>(sizes.columns));
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw Error("cuBLAS gemm failed");
        }
    }
}

} // namespace
#endif

void matmul(DType dtype, const std::byte* a, const std::byte* b, std::byte* out,
            const MatmulSizes& sizes) {
#if defined(KAEL_USE_CUBLAS)
    if (dtype == DType::F32) {
        cublas_multiply<float>(cublasSgemm, reinterpret_cast<const float*>(a),
                               reinterpret_cast<const float*>(b), reinterpret_cast<float*>(out),
                               sizes);
        return;
    }
    if (dtype == DType::F64) {
        cublas_multiply<double>(cublasDgemm, reinterpret_cast<const double*>(a),
                                reinterpret_cast<const double*>(b), reinterpret_cast<double*>(out),
                                sizes);
        return;
    }
#endif
    const std::int64_t tile_rows = (sizes.rows + tasks::kMatmulTile - 1) / tasks::kMatmulTile;
    const std::int64_t tile_columns = (sizes.columns + tasks::kMatmulTile - 1) / tasks::kMatmulTile;
    const std::int64_t tiles = sizes.batches * tile_rows * tile_columns;
    with_type(dtype, [&](auto sample) {
        using T = decltype(sample);
        for_each(tiles, tasks::MatmulTiles<T>{reinterpret_cast<const T*>(a),
                                              reinterpret_cast<const T*>(b),
                                              reinterpret_cast<T*>(out), sizes, tile_rows,
                                              tile_columns});
    });
}

} // namespace kael::device
