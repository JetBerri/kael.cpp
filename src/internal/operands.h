#pragma once

#include <utility>

#include "kael/operand.h"

namespace kael::detail {

std::pair<Tensor, Tensor> resolve_operands(const Operand& a, const Operand& b);

} // namespace kael::detail
