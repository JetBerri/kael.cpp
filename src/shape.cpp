#include <limits>
#include "kael/shape.h"
#include "kael/error.h"

namespace kael{
    
    std::string to_string(const Shape& shape){

        std::string s = "["; // initialize

        for (std::size_t i = 0; i < shape.size(); i++){
            if (i != 0) {s += ", ";} // separator only between elements, so no trailing comma
            s += std::to_string(shape[i]); 
        }

        s += "]"; 
        return s; 

    }

    void validate_shape(const Shape& shape){

        for (std::size_t i = 0; i < shape.size(); i++){
            KAEL_CHECK(shape[i] >= 0, "negative dimension: " << shape[i] << " at position " << i << " in shape: " << kael::to_string(shape));
        }

    }

    std::int64_t numel(const Shape& shape){

        validate_shape(shape);

        for (std::size_t i = 0; i < shape.size(); i++){
            if (shape[i] == 0){
                return 0;
            }
        }

        const std::int64_t max = std::numeric_limits<std::int64_t>::max();
        std::int64_t total = 1;

        for (std::size_t j = 0; j < shape.size(); j++){
            std::int64_t dim = shape[j];
            KAEL_CHECK(total <= max / dim, "number of elements overflows int64 for shape " << to_string(shape));
            total *= dim;
        }

        return total;

    }

    Strides contiguous_strides(const Shape& shape){
        
        validate_shape(shape);

        Strides result(shape.size());
        std::int64_t acc = 1;
        
        for (std::int64_t i = static_cast<std::int64_t>(shape.size()) -1; i >= 0; i--){
            const auto idx = static_cast<std::size_t>(i);
            result[idx] = acc;
            acc *= std::max<std::int64_t>(shape[idx], 1);
        }

        return result;

    }

    namespace {

        // to_string for spans (shape, strides or index) in error messages
        std::string to_string(std::span<const std::int64_t> values){
            return kael::to_string(Shape(values.begin(), values.end()));
        }

    }

    std::int64_t offset_from_index(std::span<const std::int64_t> shape,
                                   std::span<const std::int64_t> strides,
                                   std::int64_t base_offset,
                                   std::span<const std::int64_t> index){

        KAEL_CHECK(shape.size() == strides.size(), "shape and strides have different number of dimensions: shape " << to_string(shape) << " has " << shape.size() << ", strides " << to_string(strides) << " has " << strides.size());
        KAEL_CHECK(index.size() == shape.size(), "index and shape have different number of dimensions: index " << to_string(index) << " has " << index.size() << ", shape " << to_string(shape) << " has " << shape.size());

        std::int64_t offset = base_offset;

        for (std::size_t i = 0; i < shape.size(); i++){
            KAEL_CHECK(index[i] >= 0 && index[i] < shape[i], "index " << index[i] << " out of range [0, " << shape[i] << ") at dimension " << i << " in shape " << to_string(shape));
            offset += index[i] * strides[i];
        }

        return offset;

    }

}