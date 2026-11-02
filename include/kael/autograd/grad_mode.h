#pragma once

namespace kael::autograd {

class GradMode {
public:
    [[nodiscard]] static bool is_enabled() noexcept;
    static void set_enabled(bool enabled) noexcept;
};

class GradModeGuard {
public:
    explicit GradModeGuard(bool enabled) noexcept : previous_(GradMode::is_enabled()) {
        GradMode::set_enabled(enabled);
    }
    ~GradModeGuard() { GradMode::set_enabled(previous_); }

    GradModeGuard(const GradModeGuard&) = delete;
    GradModeGuard& operator=(const GradModeGuard&) = delete;

private:
    bool previous_;
};

class NoGradGuard : public GradModeGuard {
public:
    NoGradGuard() noexcept : GradModeGuard(false) {}
};

class EnableGradGuard : public GradModeGuard {
public:
    EnableGradGuard() noexcept : GradModeGuard(true) {}
};

template <typename F> auto without_grad(F compute) {
    const NoGradGuard guard;
    return compute();
}

} // namespace kael::autograd
