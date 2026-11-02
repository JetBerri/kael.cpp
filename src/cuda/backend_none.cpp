#include "kael/device_kernels.h"
#include "kael/error.h"

namespace kael::device {

namespace {

[[noreturn]] void fail_without_cuda() {
    throw Error("kael was built without CUDA; configure with KAEL_ENABLE_CUDA=ON");
}

} // namespace

bool available() {
    return false;
}

void synchronize() {}

std::byte* allocate(std::int64_t) {
    fail_without_cuda();
}

void release(std::byte*) noexcept {}

void copy_to_device(std::byte*, const std::byte*, std::int64_t) {
    fail_without_cuda();
}

void copy_to_host(std::byte*, const std::byte*, std::int64_t) {
    fail_without_cuda();
}

void copy_within_device(std::byte*, const std::byte*, std::int64_t) {
    fail_without_cuda();
}

void fill(DType, std::byte*, std::int64_t, const void*) {
    fail_without_cuda();
}

void convert(DType, DType, const ElementLayout&) {
    fail_without_cuda();
}

void elementwise(ElementOp, DType, DType, const ElementLayout&, ElementParameters) {
    fail_without_cuda();
}

void reduce(ReduceOp, DType, DType, std::byte*, const std::byte*, std::int64_t, std::int64_t) {
    fail_without_cuda();
}

void matmul(DType, const std::byte*, const std::byte*, std::byte*, const MatmulSizes&) {
    fail_without_cuda();
}

void gather(DType, std::byte*, const std::byte*, const std::int64_t*, const IndexLayout&) {
    fail_without_cuda();
}

void scatter_add(DType, std::byte*, const std::int64_t*, const std::byte*, const IndexLayout&) {
    fail_without_cuda();
}

} // namespace kael::device
