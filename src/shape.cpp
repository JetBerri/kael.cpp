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

}