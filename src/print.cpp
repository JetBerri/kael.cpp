#include <iomanip>
#include <ostream>
#include <sstream>
#include <type_traits>

#include "kael/tensor.h"

namespace kael {

namespace {

constexpr std::int64_t kSummarizeThreshold = 1000;
constexpr std::int64_t kSummarizeEdge = 3;
constexpr int kFloatPrecision = 4;

// Writes the element at logical position `linear`. Slow on purpose: going
// through the index works for any strides, and printing does not need speed.
template <typename T>
void print_element(std::ostream& os, const Tensor& t, std::int64_t linear) {
    const Shape index = index_from_linear(linear, t.shape());
    const std::int64_t pos = offset_from_index(t.shape(), t.strides(), t.offset(), index);
    const Storage& storage = *t.impl()->storage;
    const T value = reinterpret_cast<const T*>(storage.data())[pos];
    if constexpr (std::is_same_v<T, std::uint8_t>) {
        os << (value != 0 ? "true" : "false");
    } else {
        os << value;
    }
}

} // namespace

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    // A local stream, so the caller's formatting flags are left untouched.
    std::ostringstream out;
    out << std::fixed << std::setprecision(kFloatPrecision);

    const std::int64_t n = t.numel();
    const bool summarize = n > kSummarizeThreshold;

    out << "tensor([";
    dispatch(t.dtype(), [&]<typename T>() {
        for (std::int64_t i = 0; i < n; ++i) {
            if (i != 0) {
                out << ", ";
            }
            if (summarize && i == kSummarizeEdge) {
                out << "..., ";
                i = n - kSummarizeEdge;
            }
            print_element<T>(out, t, i);
        }
    });
    out << "], shape=" << to_string(t.shape()) << ", dtype=" << t.dtype() << ")";

    return os << out.str();
}

} // namespace kael
