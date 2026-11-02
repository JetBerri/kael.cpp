#include "kael/autograd/grad_mode.h"

namespace kael::autograd {

namespace {

thread_local bool grad_enabled = true;

} // namespace

bool GradMode::is_enabled() noexcept {
    return grad_enabled;
}

void GradMode::set_enabled(bool enabled) noexcept {
    grad_enabled = enabled;
}

} // namespace kael::autograd
