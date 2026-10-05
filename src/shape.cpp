#include "kael/shape.h"

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
}