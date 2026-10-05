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

}