// === SOLUTION ===
// Reference-Counted Shared Pointer (SharedPtr / WeakPtr)
// Compile: g++ -std=c++17 -Wall -Werror -O2 main.cpp

#include <atomic>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <thread>
#include <utility>
#include <vector>

namespace detail {

// One control block is shared by every SharedPtr and WeakPtr to an object.
// `weak` counts the weak references *plus one* collectively owned by all the
// strong references, so the block itself outlives the object exactly as long
// as some WeakPtr still needs to read `strong`.
struct ControlBlock {
    std::atomic<long> strong{1};
    std::atomic<long> weak{1};

    virtual ~ControlBlock() = default;
    virtual void destroy() noexcept = 0;

    void add_strong() noexcept { strong.fetch_add(1, std::memory_order_relaxed); }
    void add_weak() noexcept { weak.fetch_add(1, std::memory_order_relaxed); }

    void release_strong() noexcept {
        // acq_rel so the destructor happens-after every other thread's writes.
        if (strong.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            destroy();
            release_weak();
        }
    }

    void release_weak() noexcept {
        if (weak.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            delete this;
        }
    }
};

template <typename T>
struct PtrControlBlock final : ControlBlock {
    T* ptr;
    explicit PtrControlBlock(T* p) noexcept : ptr(p) {}
    void destroy() noexcept override {
        delete ptr;
        ptr = nullptr;
    }
};

}  // namespace detail

template <typename T>
class WeakPtr;

template <typename T>
class SharedPtr {
public:
    SharedPtr() noexcept : ptr_(nullptr), ctrl_(nullptr) {}
    SharedPtr(std::nullptr_t) noexcept : SharedPtr() {}

    explicit SharedPtr(T* p) : ptr_(p), ctrl_(nullptr) {
        if (p) {
            // If the control block allocation throws we must not leak `p`.
            try {
                ctrl_ = new detail::PtrControlBlock<T>(p);
            } catch (...) {
                delete p;
                throw;
            }
        }
    }

    SharedPtr(const SharedPtr& other) noexcept
        : ptr_(other.ptr_), ctrl_(other.ctrl_) {
        if (ctrl_) ctrl_->add_strong();
    }

    SharedPtr(SharedPtr&& other) noexcept : ptr_(other.ptr_), ctrl_(other.ctrl_) {
        other.ptr_ = nullptr;
        other.ctrl_ = nullptr;
    }

    // Taking the argument by value covers copy- and move-assignment at once and
    // is self-assignment safe: the copy is made before *this is touched.
    SharedPtr& operator=(SharedPtr other) noexcept {
        swap(other);
        return *this;
    }

    ~SharedPtr() {
        if (ctrl_) ctrl_->release_strong();
    }

    void swap(SharedPtr& other) noexcept {
        std::swap(ptr_, other.ptr_);
        std::swap(ctrl_, other.ctrl_);
    }

    void reset() noexcept { SharedPtr().swap(*this); }
    void reset(T* p) { SharedPtr(p).swap(*this); }

    T* get() const noexcept { return ptr_; }
    T& operator*() const noexcept { return *ptr_; }
    T* operator->() const noexcept { return ptr_; }

    long use_count() const noexcept {
        return ctrl_ ? ctrl_->strong.load(std::memory_order_relaxed) : 0;
    }

    explicit operator bool() const noexcept { return ptr_ != nullptr; }

private:
    friend class WeakPtr<T>;

    // Used by WeakPtr::lock() once it has already bumped the strong count.
    SharedPtr(T* p, detail::ControlBlock* c) noexcept : ptr_(p), ctrl_(c) {}

    T* ptr_;
    detail::ControlBlock* ctrl_;
};

template <typename T>
class WeakPtr {
public:
    WeakPtr() noexcept : ptr_(nullptr), ctrl_(nullptr) {}

    WeakPtr(const SharedPtr<T>& s) noexcept : ptr_(s.ptr_), ctrl_(s.ctrl_) {
        if (ctrl_) ctrl_->add_weak();
    }

    WeakPtr(const WeakPtr& other) noexcept : ptr_(other.ptr_), ctrl_(other.ctrl_) {
        if (ctrl_) ctrl_->add_weak();
    }

    WeakPtr(WeakPtr&& other) noexcept : ptr_(other.ptr_), ctrl_(other.ctrl_) {
        other.ptr_ = nullptr;
        other.ctrl_ = nullptr;
    }

    WeakPtr& operator=(WeakPtr other) noexcept {
        swap(other);
        return *this;
    }

    ~WeakPtr() {
        if (ctrl_) ctrl_->release_weak();
    }

    void swap(WeakPtr& other) noexcept {
        std::swap(ptr_, other.ptr_);
        std::swap(ctrl_, other.ctrl_);
    }

    void reset() noexcept { WeakPtr().swap(*this); }

    long use_count() const noexcept {
        return ctrl_ ? ctrl_->strong.load(std::memory_order_relaxed) : 0;
    }

    bool expired() const noexcept { return use_count() == 0; }

    // A plain load-then-increment would race with the last SharedPtr dying in
    // between; the CAS loop only claims a reference while one still exists.
    SharedPtr<T> lock() const noexcept {
        if (!ctrl_) return SharedPtr<T>();

        long cur = ctrl_->strong.load(std::memory_order_relaxed);
        while (cur != 0) {
            if (ctrl_->strong.compare_exchange_weak(cur, cur + 1,
                                                    std::memory_order_acq_rel,
                                                    std::memory_order_relaxed)) {
                return SharedPtr<T>(ptr_, ctrl_);
            }
        }
        return SharedPtr<T>();
    }

private:
    T* ptr_;
    detail::ControlBlock* ctrl_;
};

template <typename T, typename... Args>
SharedPtr<T> MakeShared(Args&&... args) {
    return SharedPtr<T>(new T(std::forward<Args>(args)...));
}

// --- Test harness ---

struct Tracked {
    static int alive;
    int value;

    explicit Tracked(int v = 0) : value(v) { ++alive; }
    ~Tracked() { --alive; }
};

int Tracked::alive = 0;

int main() {
    {
        // Basic ownership and copy semantics
        SharedPtr<Tracked> a(new Tracked(42));
        assert(a.use_count() == 1);
        assert(Tracked::alive == 1);
        assert(a->value == 42);
        assert((*a).value == 42);

        {
            SharedPtr<Tracked> b = a;
            assert(a.use_count() == 2);
            assert(b.get() == a.get());
        }
        assert(a.use_count() == 1);
        assert(Tracked::alive == 1);
    }
    assert(Tracked::alive == 0);

    {
        // Move leaves the source empty and does not touch the count
        SharedPtr<Tracked> a = MakeShared<Tracked>(7);
        SharedPtr<Tracked> b = std::move(a);
        assert(!a);
        assert(a.use_count() == 0);
        assert(b.use_count() == 1);
        assert(b->value == 7);
    }
    assert(Tracked::alive == 0);

    {
        // WeakPtr observes without owning
        SharedPtr<Tracked> a(new Tracked(1));
        WeakPtr<Tracked> w(a);
        assert(a.use_count() == 1);
        assert(!w.expired());

        {
            SharedPtr<Tracked> locked = w.lock();
            assert(locked);
            assert(a.use_count() == 2);
        }
        assert(a.use_count() == 1);

        a.reset();
        assert(Tracked::alive == 0);
        assert(w.expired());
        assert(!w.lock());
    }

    {
        // A live WeakPtr keeps the control block alive after the object dies
        WeakPtr<Tracked> w;
        {
            SharedPtr<Tracked> a(new Tracked(5));
            w = WeakPtr<Tracked>(a);
        }
        assert(w.expired());
        assert(w.use_count() == 0);
    }
    assert(Tracked::alive == 0);

    {
        // Self-assignment must not destroy the object
        SharedPtr<Tracked> a(new Tracked(3));
        SharedPtr<Tracked>& alias = a;
        a = alias;
        assert(a.use_count() == 1);
        assert(a->value == 3);
    }
    assert(Tracked::alive == 0);

    {
        // An empty SharedPtr is falsy and assignable from a live one
        SharedPtr<Tracked> a(new Tracked(9));
        SharedPtr<Tracked> b(nullptr);
        assert(!b);
        assert(b.use_count() == 0);
        b = a;
        assert(a.use_count() == 2);
    }
    assert(Tracked::alive == 0);

    {
        // Refcount updates are atomic across threads
        SharedPtr<Tracked> a(new Tracked(100));
        std::vector<std::thread> threads;
        for (int t = 0; t < 8; ++t) {
            threads.emplace_back([a]() {
                for (int i = 0; i < 1000; ++i) {
                    SharedPtr<Tracked> copy = a;
                    assert(copy->value == 100);
                }
            });
        }
        for (auto& th : threads) th.join();
        assert(a.use_count() == 1);
        assert(Tracked::alive == 1);
    }
    assert(Tracked::alive == 0);

    std::cout << "All tests passed.\n";
}
