#include "kael/parallel.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#if !defined(_WIN32)
#include <pthread.h>
#endif

#include "kael/autograd/grad_mode.h"
#include "kael/error.h"

namespace kael {

namespace {

constexpr std::int64_t kChunksPerThread = 4;

thread_local bool inside_parallel_region = false;

class Job {
public:
    Job(std::int64_t begin, std::int64_t end, std::int64_t chunk_size, const ParallelBody& body)
        : begin_(begin), end_(end), chunk_size_(chunk_size), body_(body),
          grad_enabled_(autograd::GradMode::is_enabled()) {}

    void work() noexcept {
        const autograd::GradModeGuard grad_mode(grad_enabled_);
        const bool was_inside = std::exchange(inside_parallel_region, true);
        for (;;) {
            const std::int64_t start = next_.fetch_add(chunk_size_, std::memory_order_relaxed);
            if (start >= end_) {
                break;
            }
            try {
                body_(start, std::min(start + chunk_size_, end_));
            } catch (...) {
                keep_first_error(std::current_exception());
                next_.store(end_, std::memory_order_relaxed);
            }
        }
        inside_parallel_region = was_inside;
    }

    void rethrow_error() const {
        if (error_) {
            std::rethrow_exception(error_);
        }
    }

    void start() noexcept { next_.store(begin_, std::memory_order_relaxed); }

private:
    void keep_first_error(std::exception_ptr error) {
        const std::lock_guard lock(error_mutex_);
        if (!error_) {
            error_ = std::move(error);
        }
    }

    std::int64_t begin_;
    std::int64_t end_;
    std::int64_t chunk_size_;
    const ParallelBody& body_;
    bool grad_enabled_;
    std::atomic<std::int64_t> next_{0};
    std::mutex error_mutex_;
    std::exception_ptr error_;
};

class ThreadPool {
public:
    explicit ThreadPool(std::int64_t worker_count) {
        workers_.reserve(static_cast<std::size_t>(worker_count));
        for (std::int64_t i = 0; i < worker_count; ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    ~ThreadPool() {
        {
            const std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        wake_workers_.notify_all();
        for (std::thread& worker : workers_) {
            worker.join();
        }
    }

    void run(Job& job) {
        {
            const std::lock_guard lock(mutex_);
            job_ = &job;
            ++generation_;
            busy_workers_ = workers_.size();
        }
        wake_workers_.notify_all();
        job.work();
        std::unique_lock lock(mutex_);
        workers_done_.wait(lock, [this] { return busy_workers_ == 0; });
        job_ = nullptr;
    }

private:
    void worker_loop() {
        std::uint64_t seen_generation = 0;
        std::unique_lock lock(mutex_);
        for (;;) {
            wake_workers_.wait(lock, [&] { return stopping_ || generation_ != seen_generation; });
            if (stopping_) {
                return;
            }
            seen_generation = generation_;
            Job* job = job_;
            lock.unlock();
            job->work();
            lock.lock();
            if (--busy_workers_ == 0) {
                workers_done_.notify_one();
            }
        }
    }

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable wake_workers_;
    std::condition_variable workers_done_;
    Job* job_ = nullptr;
    std::uint64_t generation_ = 0;
    std::size_t busy_workers_ = 0;
    bool stopping_ = false;
};

// Efficiency cores slow matmul and elementwise kernels down, so Apple Silicon uses only the others.
std::int64_t hardware_thread_count() {
#if defined(__APPLE__)
    int performance_cores = 0;
    std::size_t size = sizeof(performance_cores);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &performance_cores, &size, nullptr, 0) == 0 &&
        performance_cores > 0) {
        return performance_cores;
    }
#endif
    return std::max<std::int64_t>(1, std::thread::hardware_concurrency());
}

std::int64_t default_thread_count() {
    if (const char* configured = std::getenv("KAEL_NUM_THREADS")) {
        char* end = nullptr;
        const std::int64_t parsed = std::strtoll(configured, &end, 10);
        if (parsed > 0 && *end == '\0') {
            return parsed;
        }
    }
    return hardware_thread_count();
}

class ParallelRuntime {
public:
    static ParallelRuntime& instance() {
        static ParallelRuntime runtime;
        return runtime;
    }

#if !defined(_WIN32)
    ParallelRuntime() {
        pthread_atfork(nullptr, nullptr, [] { instance().forget_threads_after_fork(); });
    }
#endif

    [[nodiscard]] std::int64_t thread_count() const noexcept {
        return thread_count_.load(std::memory_order_relaxed);
    }

    void set_thread_count(std::int64_t count) {
        const std::lock_guard lock(mutex_);
        thread_count_.store(count, std::memory_order_relaxed);
        pool_.reset();
    }

    bool try_run(Job& job) {
        const std::unique_lock lock(mutex_, std::try_to_lock);
        if (!lock.owns_lock()) {
            return false;
        }
        if (!pool_ && thread_count() > 1) {
            pool_ = std::make_unique<ThreadPool>(thread_count() - 1);
        }
        if (!pool_) {
            return false;
        }
        pool_->run(job);
        return true;
    }

private:
#if !defined(_WIN32)
    // The workers do not exist in a forked child, so the pool can be neither used nor destroyed.
    void forget_threads_after_fork() {
        new (&mutex_) std::mutex;
        [[maybe_unused]] const auto* forgotten = pool_.release();
        inside_parallel_region = false;
    }
#endif

    std::mutex mutex_;
    std::atomic<std::int64_t> thread_count_{default_thread_count()};
    std::unique_ptr<ThreadPool> pool_;
};

} // namespace

void set_num_threads(std::int64_t count) {
    KAEL_CHECK_VALUE(count > 0, "thread count must be positive, got " << count);
    KAEL_CHECK(!inside_parallel_region, "cannot change the thread count inside parallel_for");
    ParallelRuntime::instance().set_thread_count(count);
}

std::int64_t get_num_threads() {
    return ParallelRuntime::instance().thread_count();
}

void parallel_for(std::int64_t begin, std::int64_t end, std::int64_t grain,
                  const ParallelBody& body) {
    KAEL_CHECK_VALUE(grain > 0, "parallel_for grain must be positive, got " << grain);
    const std::int64_t length = end - begin;
    if (length <= 0) {
        return;
    }
    const std::int64_t threads = inside_parallel_region ? 1 : get_num_threads();
    if (length <= grain || threads == 1) {
        body(begin, end);
        return;
    }
    const std::int64_t chunks = threads * kChunksPerThread;
    const std::int64_t chunk_size = std::max(grain, (length + chunks - 1) / chunks);
    Job job(begin, end, chunk_size, body);
    job.start();
    if (ParallelRuntime::instance().try_run(job)) {
        job.rethrow_error();
    } else {
        body(begin, end);
    }
}

} // namespace kael
