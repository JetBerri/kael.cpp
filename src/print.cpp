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

template <typename T> void print_element(std::ostream& os, const Tensor& t, std::int64_t linear) {
    const T value = t.at<T>(index_from_linear(linear, t.shape()));
    if constexpr (std::is_same_v<T, std::uint8_t>) {
        os << (value != 0 ? "true" : "false");
    } else {
        os << value;
    }
}

template <typename T> void print_elements(std::ostream& os, const Tensor& t) {
    const std::int64_t count = t.numel();
    const bool summarize = count > kSummarizeThreshold;
    for (std::int64_t i = 0; i < count; ++i) {
        if (i != 0) {
            os << ", ";
        }
        if (summarize && i == kSummarizeEdge) {
            os << "..., ";
            i = count - kSummarizeEdge;
        }
        print_element<T>(os, t, i);
    }
}

} // namespace

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    std::ostringstream text;
    text << std::fixed << std::setprecision(kFloatPrecision) << "tensor([";
    dispatch(t.dtype(), [&]<typename T>() { print_elements<T>(text, t); });
    text << "], shape=" << to_string(t.shape()) << ", dtype=" << t.dtype() << ")";
    return os << text.str();
}

} // namespace kael
