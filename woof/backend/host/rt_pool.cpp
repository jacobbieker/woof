// The host backend's runtime (WOOF_BACKEND=host): one pool of worker
// threads that runs a launch's blocks, and the per-worker fibers that run a
// block's threads when its kernel synchronises (rt.h).  Built once per
// compiler and cached (woof/backend/host/loader.py); every unit library
// links against it, so a process has one pool however many units it loads.
//
// A launch is a parallel loop over its blocks: each worker takes the next
// block index from an atomic counter until none is left.  Blocks are
// independent in CUDA (no grid-wide barrier exists outside cooperative
// launches, which woof does not use), so any order is a valid schedule.
// Launches are serialised, as on one CUDA stream.
//
// Fibers: a block whose kernel calls __syncthreads, __syncwarp or a warp
// collective runs each of its threads as a fiber on the worker that owns the
// block.  Switching is a few instructions (x86-64 below; any other machine
// uses ucontext).  A fiber's stack is mapped once per worker and reused.
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>
#if !defined(__x86_64__)
#include <ucontext.h>
#endif

namespace {

size_t env_bytes(const char *name, size_t fallback_kib)
{
    const char *v = std::getenv(name);
    size_t kib = fallback_kib;
    if (v != nullptr && *v != '\0') {
        long long n = std::atoll(v);
        if (n > 0) kib = (size_t)n;
    }
    return kib * 1024;
}

// ---------------------------------------------------------------- errors

std::mutex error_lock;
std::string error_text;
std::atomic<int> error_flag{0};

}  // namespace

extern "C" void woof_rt_fail(const char *message)
{
    std::lock_guard<std::mutex> hold(error_lock);
    if (!error_flag.load()) {
        error_text = message ? message : "kernel failure";
        error_flag.store(1);
    }
}

extern "C" int woof_rt_failed() { return error_flag.load(std::memory_order_relaxed); }

// Copies the first failure into ``out`` and clears it; returns 1 if there was one.
extern "C" int woof_rt_take_error(char *out, int size)
{
    std::lock_guard<std::mutex> hold(error_lock);
    if (!error_flag.load()) return 0;
    if (out != nullptr && size > 0) {
        std::strncpy(out, error_text.c_str(), (size_t)size - 1);
        out[size - 1] = '\0';
    }
    error_text.clear();
    error_flag.store(0);
    return 1;
}

// ---------------------------------------------------------------- fibers

struct woof_fiber {
    void *stack = nullptr;
    size_t size = 0;
#if defined(__x86_64__)
    void *sp = nullptr;
#else
    ucontext_t uc;
#endif
};

namespace {

struct WorkerFibers {
#if defined(__x86_64__)
    void *home_sp = nullptr;
#else
    ucontext_t home;
#endif
    std::vector<woof_fiber *> fibers;
    ~WorkerFibers()
    {
        for (woof_fiber *f : fibers) {
            munmap(f->stack, f->size);
            delete f;
        }
    }
};

thread_local WorkerFibers worker_fibers;

}  // namespace

#if defined(__x86_64__)
// woof_rt_switch(save, to): save the callee-saved registers on this stack,
// store its pointer in *save, and continue on the stack ``to``.
extern "C" void woof_rt_switch(void **save, void *to);
extern "C" void woof_rt_trampoline();
asm(R"(
    .text
    .globl woof_rt_switch
    .type woof_rt_switch, @function
woof_rt_switch:
    pushq %rbp
    pushq %rbx
    pushq %r12
    pushq %r13
    pushq %r14
    pushq %r15
    movq %rsp, (%rdi)
    movq %rsi, %rsp
    popq %r15
    popq %r14
    popq %r13
    popq %r12
    popq %rbx
    popq %rbp
    ret
    .size woof_rt_switch, .-woof_rt_switch

    .globl woof_rt_trampoline
    .type woof_rt_trampoline, @function
woof_rt_trampoline:
    movq %r12, %rdi
    callq *%r13
    ud2
    .size woof_rt_trampoline, .-woof_rt_trampoline
    .section .note.GNU-stack,"",@progbits
    .text
)");
#endif

// The ``index``-th fiber of the calling worker, its stack mapped on first use.
extern "C" woof_fiber *woof_rt_fiber(int index)
{
    WorkerFibers &w = worker_fibers;
    while ((int)w.fibers.size() <= index) {
        static const size_t size = env_bytes("WOOF_HOST_FIBER_STACK_KB", 4096);
        const size_t page = (size_t)sysconf(_SC_PAGESIZE);
        void *base = mmap(nullptr, size + page, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (base == MAP_FAILED) return nullptr;
        mprotect(base, page, PROT_NONE);          // guard page below the stack
        woof_fiber *f = new woof_fiber;
        f->stack = base;
        f->size = size + page;
        w.fibers.push_back(f);
    }
    return w.fibers[(size_t)index];
}

// Make ``f`` start ``entry(arg)`` at its next resume.  ``entry`` must never
// return: it ends by yielding for the last time.
extern "C" void woof_rt_fiber_start(woof_fiber *f, void (*entry)(void *), void *arg)
{
#if defined(__x86_64__)
    uintptr_t top = ((uintptr_t)f->stack + f->size) & ~(uintptr_t)15;
    void **sp = (void **)top;
    *--sp = (void *)woof_rt_trampoline;   // woof_rt_switch's ret
    *--sp = nullptr;                      // rbp
    *--sp = nullptr;                      // rbx
    *--sp = arg;                          // r12
    *--sp = (void *)entry;                // r13
    *--sp = nullptr;                      // r14
    *--sp = nullptr;                      // r15
    f->sp = sp;
#else
    getcontext(&f->uc);
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    f->uc.uc_stack.ss_sp = (char *)f->stack + page;
    f->uc.uc_stack.ss_size = f->size - page;
    f->uc.uc_link = nullptr;
    makecontext(&f->uc, (void (*)())entry, 1, arg);
#endif
}

// From the worker's own context into ``f``; returns when ``f`` yields.
extern "C" void woof_rt_resume(woof_fiber *f)
{
#if defined(__x86_64__)
    woof_rt_switch(&worker_fibers.home_sp, f->sp);
#else
    swapcontext(&worker_fibers.home, &f->uc);
#endif
}

// From fiber ``f`` back to its worker's context.
extern "C" void woof_rt_yield(woof_fiber *f)
{
#if defined(__x86_64__)
    woof_rt_switch(&f->sp, worker_fibers.home_sp);
#else
    swapcontext(&f->uc, &worker_fibers.home);
#endif
}

// ---------------------------------------------------------------- the pool

namespace {

typedef void (*BlockFn)(void *ctx, unsigned long long block);

struct Pool {
    std::mutex launch;                  // one launch at a time (a stream)
    std::mutex m;
    std::condition_variable wake, done;
    std::vector<pthread_t> threads;
    unsigned long long generation = 0;
    BlockFn fn = nullptr;
    void *ctx = nullptr;
    unsigned long long nblocks = 0;
    std::atomic<unsigned long long> next{0};
    int wanted = 0;                     // workers taking part in this launch
    int running = 0;
};

Pool *pool()
{
    static Pool *p = new Pool;          // never destroyed: workers outlive exit
    return p;
}

void *worker_main(void *arg)
{
    Pool *p = pool();
    const int me = (int)(intptr_t)arg;
    unsigned long long seen = 0;
    for (;;) {
        BlockFn fn;
        void *ctx;
        unsigned long long n;
        {
            std::unique_lock<std::mutex> hold(p->m);
            p->wake.wait(hold, [&] { return p->generation != seen && me < p->wanted; });
            seen = p->generation;
            fn = p->fn;
            ctx = p->ctx;
            n = p->nblocks;
        }
        for (;;) {
            unsigned long long b = p->next.fetch_add(1, std::memory_order_relaxed);
            if (b >= n) break;
            fn(ctx, b);
        }
        std::lock_guard<std::mutex> hold(p->m);
        if (--p->running == 0) p->done.notify_all();
    }
    return nullptr;
}

bool grow(Pool *p, int count)
{
    static const size_t stack = env_bytes("WOOF_HOST_THREAD_STACK_KB", 65536);
    while ((int)p->threads.size() < count) {
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, stack);
        pthread_t t;
        int rc = pthread_create(&t, &attr, worker_main, (void *)(intptr_t)p->threads.size());
        pthread_attr_destroy(&attr);
        if (rc != 0) return false;
        pthread_detach(t);
        p->threads.push_back(t);
    }
    return true;
}

}  // namespace

// Run fn(ctx, b) for every block b < nblocks on up to ``workers`` threads;
// returns when all have finished.  Nonzero if the pool could not start.
extern "C" int woof_rt_run(unsigned long long nblocks, int workers, BlockFn fn, void *ctx)
{
    if (nblocks == 0) return 0;
    Pool *p = pool();
    std::lock_guard<std::mutex> serial(p->launch);
    int count = workers < 1 ? 1 : workers;
    if ((unsigned long long)count > nblocks) count = (int)nblocks;
    {
        std::lock_guard<std::mutex> hold(p->m);
        if (!grow(p, count)) {
            woof_rt_fail("the host backend could not start its worker threads");
            return 1;
        }
        p->fn = fn;
        p->ctx = ctx;
        p->nblocks = nblocks;
        p->next.store(0);
        p->wanted = count;
        p->running = count;
        ++p->generation;
    }
    p->wake.notify_all();
    std::unique_lock<std::mutex> hold(p->m);
    p->done.wait(hold, [&] { return p->running == 0; });
    p->wanted = 0;
    return 0;
}
