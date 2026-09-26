/*
SINGLE-FILE C++20 MATCHING ENGINE -- custom containers, C-style data layout
=============================================================================
Build (GCC or Clang, Linux/macOS with a C++20 standard library):
  g++ -std=c++20 -O3 -DNDEBUG -march=native -pthread matching_engine.cpp -o engine
  ./engine --demo
  ./engine --test
  ./engine --bench 1000000
  ./engine --cli
Debug / memory and undefined-behavior checking:
  g++ -std=c++20 -O1 -g -pthread -fsanitize=address,undefined \
      -fno-omit-frame-pointer matching_engine.cpp -o engine_debug
  ./engine_debug --test
Thread checking (separate binary, where ThreadSanitizer is supported):
  g++ -std=c++20 -O1 -g -pthread -fsanitize=thread matching_engine.cpp -o engine_tsan
  ./engine_tsan --queue-test
Embed without the demonstration main(): define HFT_NO_MAIN before inclusion.
This is C++, not a C-compatible translation unit. No third-party dependencies.

VALIDATION OF THIS DELIVERABLE (2026-09-09)
  GCC 13: release build with -Wall -Wextra -Wpedantic -Wconversion -Wshadow
  -Werror; standalone and HFT_NO_MAIN builds passed. Deterministic tests,
  300,000 randomized command-by-command comparisons against the slow oracle,
  1,000,000-message concurrent SPSC test, and output backpressure test passed.
  The full suite also passed AddressSanitizer + UndefinedBehaviorSanitizer.
  LeakSanitizer could not run under this runtime's process tracing; the rerun
  used ASAN_OPTIONS=detect_leaks=0 (memory/UB checks remained enabled).
  ThreadSanitizer --queue-test passed. CLI and demo were exercised. These
  checks are evidence of testing, not a formal proof or production readiness.

CONTENTS
  1. FixedVector: fixed-capacity POD sequence; no allocation or growth.
  2. IdHash: bounded linear probes and backward-shift deletion.
  3. Bitmap: recursive occupancy summaries, ascending/descending price lookup.
  4. OrderPool: preallocated nodes; index 0 is the null link.
  5. SpscRing: acquire/release queue; exactly one producer and one consumer.
  6. OrderBook: price-time priority, executions, cancel, replace, depth views.
  7. QueuedEngine: bounded ingress/egress with explicit backpressure.
  8. Independent slow reference, deterministic/randomized/concurrent tests.
  9. CLI, worked demo, narrowly scoped benchmark.

CONTRACT AND MATCHING POLICY
  * One instrument per book. One thread exclusively owns/mutates the book.
    Feed multiple instruments to separate owners; externally sequence multiple
    gateways. Concurrent direct reads of the mutable book are NOT supported.
  * Prices are signed integer ticks, in [BaseTick, BaseTick + PriceCount - 1].
    Convert decimal wire prices exactly before calling this API. No doubles in
    matching, and no currency/notional arithmetic in this engine.
  * IDs and quantities must be nonzero. IDs are unique among LIVE orders only;
    historical duplicate detection belongs to an external sequencer/session.
  * Execution price is the resting maker's price. FIFO within each price.
  * Limit: GTC, IOC, FOK, PostOnly. Market: IOC or FOK only (tick ignored).
    IOC residual expires. FOK rejects with no executions unless fully fillable.
    PostOnly rejects a crossing order; it never slides/reprices the order.
  * Replace specifies NEW REMAINING quantity, NOT lifetime total quantity.
    Same-price reduction/equality retains priority. Increase or price change
    loses priority and may trade. Replacement keeps ID and side; it uses GTC.
    Replace quantity zero means cancel. Invalid replacements preserve the old
    order. An existing pool slot is reused, so replacement works at capacity.
  * A full pool accepts a GTC order if it fully fills or matching frees a slot
    for its residual. Otherwise reject BEFORE any fill. IOC/FOK need no slot.
  * Every processed command emits zero or more Trade events then exactly one
    Done or Rejected event. All events carry the command sequence number.
    For Add and priority-losing Replace: Done.qty = executed quantity,
    Done.leaves = resting quantity, Done.canceled = expired IOC residual.
    For Cancel or priority-preserving Replace: canceled = removed quantity.
    Rejected events have zero qty/leaves/canceled; they do not change the book.
  * Output is a caller-owned EventBatch of MaxOrders+1 entries. At most one
    trade per previously resting order plus one result: output cannot overflow.
    No callbacks, allocations, syscalls, locks or printing in book operations.
    QueuedEngine reserves that worst-case output space before taking input;
    this conservative policy avoids partial event publication failures.
  * No self-trade prevention, auctions, iceberg/stop/peg orders, risk checks,
    session protocol, persistence/recovery, replication, or exchange-specific
    rules. This is a complete bounded matching CORE and executable study file,
    not a certified exchange system or a claim of measured HFT latency.

COMPLEXITY / LAYOUT
  Hash lookup/delete: expected O(1), worst O(HashCapacity); all loops bounded.
  FIFO unlink/append: O(1). Bitmap lookup/update: O(log_64 PriceCount).
  BBO query: O(1), cached. Sweep: O(fills + levels * log_64 PriceCount), plus
  hash deletion costs. FOK (and GTC when pool full) first scans eligible LEVEL
  totals; its preflight is separate from execution. Memory is fixed at compile
  time. Level totals are uint64_t; MaxOrders < 2^32 and each qty < 2^32 make
  their sum fit in uint64_t. There is no claim of worst-case constant latency.
  No STL CONTAINERS are used, including in tests. Standard atomics, type traits,
  bit operations, chrono and threads are used instead of reinventing primitives.
  64-byte alignment is a tuning assumption, not a universal cache-line guarantee.
  Place large books/queues in static storage or allocate ONCE before the loop.
  A dense ladder trades memory for locality; widely spaced prices may need a
  paged ladder. Benchmark your actual workload and hardware before tuning.
=============================================================================
*/

#include <atomic>
#include <bit>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>
#include <type_traits>

namespace hft {
using Id = std::uint64_t;
using Qty = std::uint32_t;
using Tick = std::int32_t;
using Index = std::uint32_t;
constexpr std::size_t CacheLine = 64;

// Internal impossible conditions must remain checked even with -DNDEBUG.
inline void require_internal(bool ok) noexcept { if (!ok) std::abort(); }
constexpr Qty smaller(Qty a, Qty b) noexcept { return a < b ? a : b; }
constexpr bool power_of_two(std::size_t n) noexcept { return n && !(n & (n-1)); }

// 1. Narrow fixed vector: intentionally only trivial element types. Unlike
// std::vector it has no allocator, resize, relocation, or nontrivial lifetimes.
template<class T, std::size_t N>
class FixedVector {
    static_assert(N > 0 && std::is_trivially_copyable_v<T>);
    T data_[N]{};
    std::size_t size_ = 0;
public:
    bool push_back(const T& v) noexcept {
        if (size_ == N) return false;
        data_[size_++] = v;
        return true;
    }
    void clear() noexcept { size_ = 0; }
    std::size_t size() const noexcept { return size_; }
    static constexpr std::size_t capacity() noexcept { return N; }
    bool empty() const noexcept { return !size_; }
    T& operator[](std::size_t i) noexcept { assert(i < size_); return data_[i]; }
    const T& operator[](std::size_t i) const noexcept { assert(i < size_); return data_[i]; }
    T* begin() noexcept { return data_; }
    T* end() noexcept { return data_ + size_; }
    const T* begin() const noexcept { return data_; }
    const T* end() const noexcept { return data_ + size_; }
};

// 2. Nonzero ID -> nonzero pool index. Empty key is 0; no tombstones.
struct IdSlot { Id key = 0; Index value = 0; std::uint32_t pad = 0; };
static_assert(sizeof(IdSlot) == 16);
template<std::size_t Capacity>
class IdHash {
    static_assert(power_of_two(Capacity) && Capacity >= 2);
    static constexpr std::size_t Mask = Capacity - 1;
    IdSlot slots_[Capacity]{};
    std::size_t size_ = 0;
public:
    static std::uint64_t hash(Id x) noexcept {
        x += 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }
    std::size_t size() const noexcept { return size_; }
    Index find(Id key) const noexcept {
        if (!key) return 0;
        std::size_t p = hash(key) & Mask;
        for (std::size_t n = 0; n < Capacity; ++n, p = (p+1) & Mask) {
            if (!slots_[p].key) return 0;
            if (slots_[p].key == key) return slots_[p].value;
        }
        return 0;
    }
    bool insert(Id key, Index value) noexcept {
        if (!key || !value) return false;
        std::size_t p = hash(key) & Mask;
        for (std::size_t n = 0; n < Capacity; ++n, p = (p+1) & Mask) {
            if (slots_[p].key == key) return false;
            if (!slots_[p].key) {
                slots_[p] = {key, value, 0}; ++size_; return true;
            }
        }
        return false;
    }
    bool erase(Id key) noexcept {
        if (!key) return false;
        std::size_t hole = hash(key) & Mask;
        bool found = false;
        for (std::size_t n = 0; n < Capacity; ++n, hole = (hole+1) & Mask) {
            if (!slots_[hole].key) return false;
            if (slots_[hole].key == key) { found = true; break; }
        }
        if (!found) return false;
        slots_[hole] = {};
        std::size_t scan = (hole+1) & Mask;
        for (std::size_t n = 0; n < Capacity-1; ++n, scan = (scan+1) & Mask) {
            if (!slots_[scan].key) break;
            const std::size_t home = hash(slots_[scan].key) & Mask;
            // Move only if lookup for this key would pass through the hole.
            if (((hole-home) & Mask) < ((scan-home) & Mask)) {
                slots_[hole] = slots_[scan];
                slots_[scan] = {};
                hole = scan;
            }
        }
        --size_; return true;
    }
};

// 3. Each summary bit says whether a child word is nonempty. Bit scans are
// always guarded by x!=0. Masks never shift by 64. Returned index -1 = absent.
template<std::size_t Bits, bool Small = (Bits <= 64)> class Bitmap;
template<std::size_t Bits>
class Bitmap<Bits, true> {
    static_assert(Bits > 0);
    std::uint64_t word_ = 0;
public:
    void set(int i) noexcept { assert(i>=0 && i<int(Bits)); word_ |= 1ULL << i; }
    void clear(int i) noexcept { assert(i>=0 && i<int(Bits)); word_ &= ~(1ULL << i); }
    bool test(int i) const noexcept { return i>=0 && i<int(Bits) && ((word_ >> i)&1); }
    int next(int start) const noexcept {
        if (start < 0) start = 0;
        if (start >= int(Bits)) return -1;
        const auto x = word_ & (~0ULL << start);
        return x ? int(std::countr_zero(x)) : -1;
    }
    int prev(int start) const noexcept {
        if (start < 0) return -1;
        if (start >= int(Bits)) start = int(Bits)-1;
        const auto x = word_ & (~0ULL >> (63-start));
        return x ? 63-int(std::countl_zero(x)) : -1;
    }
};
template<std::size_t Bits>
class Bitmap<Bits, false> {
    static_assert(Bits <= std::size_t(std::numeric_limits<int>::max()));
    static constexpr std::size_t Words = (Bits+63)/64;
    std::uint64_t words_[Words]{};
    Bitmap<Words> summary_;
public:
    void set(int i) noexcept {
        assert(i>=0 && i<int(Bits));
        const int w = i/64;
        if (!words_[w]) summary_.set(w);
        words_[w] |= 1ULL << (i%64);
    }
    void clear(int i) noexcept {
        assert(i>=0 && i<int(Bits));
        const int w = i/64;
        words_[w] &= ~(1ULL << (i%64));
        if (!words_[w]) summary_.clear(w);
    }
    bool test(int i) const noexcept {
        return i>=0 && i<int(Bits) && ((words_[i/64] >> (i%64))&1);
    }
    int next(int start) const noexcept {
        if (start < 0) start = 0;
        if (start >= int(Bits)) return -1;
        int w = start/64;
        auto x = words_[w] & (~0ULL << (start%64));
        if (x) return w*64 + int(std::countr_zero(x));
        w = summary_.next(w+1);
        return w < 0 ? -1 : w*64 + int(std::countr_zero(words_[w]));
    }
    int prev(int start) const noexcept {
        if (start < 0) return -1;
        if (start >= int(Bits)) start = int(Bits)-1;
        int w = start/64;
        auto x = words_[w] & (~0ULL >> (63-start%64));
        if (x) return w*64 + 63-int(std::countl_zero(x));
        w = summary_.prev(w-1);
        return w < 0 ? -1 : w*64 + 63-int(std::countl_zero(words_[w]));
    }
};

enum class Side : std::uint8_t { Buy, Sell };
enum class Type : std::uint8_t { Limit, Market };
enum class Tif : std::uint8_t { GTC, IOC, FOK, PostOnly };
enum class Action : std::uint8_t { Add, Cancel, Replace };
enum class EventKind : std::uint8_t { Trade, Done, Rejected };
enum class Reason : std::uint8_t {
    None, BadAction, BadId, BadSide, BadType, BadTif, BadQty,
    BadPrice, DuplicateId, UnknownId, PoolFull, WouldCross, FokUnfilled
};
struct Command {
    Action action = Action::Add;
    Side side = Side::Buy;
    Type type = Type::Limit;
    Tif tif = Tif::GTC;
    Id id = 0;
    Tick tick = 0;
    Qty qty = 0;
};
struct Event {
    std::uint64_t sequence = 0;
    Id id = 0;                  // command ID (taker for Trade)
    Id maker_id = 0;            // only meaningful for Trade
    Tick tick = 0;             // Trade execution price; otherwise request tick
    Qty qty = 0;
    Qty leaves = 0;
    Qty canceled = 0;
    Side side = Side::Buy;      // aggressor side for Trade
    EventKind kind = EventKind::Done;
    Reason reason = Reason::None;
    bool operator==(const Event&) const = default;
};
struct Order {
    Id id = 0;
    Qty qty = 0;
    std::uint32_t level = 0;   // offset in ladder, not an absolute tick
    Index prev = 0, next = 0;
    Side side = Side::Buy;
    std::uint8_t pad[7]{};
};
static_assert(sizeof(Order) == 32);
struct alignas(32) PriceLevel {
    std::uint64_t total_qty = 0;
    Index head = 0, tail = 0;
    std::uint32_t count = 0;
};
static_assert(sizeof(PriceLevel) == 32);
struct OrderView {
    Id id = 0; Qty qty = 0; Tick tick = 0; Side side = Side::Buy;
    bool operator==(const OrderView&) const = default;
};
struct LevelView {
    Tick tick = 0; std::uint64_t qty = 0; std::uint32_t count = 0;
    bool operator==(const LevelView&) const = default;
};

// 4. Free list uses the same next field as active FIFO nodes. Pool owns objects
// for its entire lifetime, avoiding placement-new and object-lifetime pitfalls.
template<std::size_t N>
class OrderPool {
    static_assert(N > 0 && N < std::numeric_limits<Index>::max());
    alignas(CacheLine) Order nodes_[N+1]{};
    Index free_ = 1;
    std::size_t available_ = N;
public:
    OrderPool() noexcept {
        for (Index i=1; i<=N; ++i) nodes_[i].next = i<N ? i+1 : 0;
    }
    Index acquire() noexcept {
        if (!free_) return 0;
        Index i = free_; free_ = nodes_[i].next;
        nodes_[i] = {}; --available_; return i;
    }
    void release(Index i) noexcept {
        assert(i>0 && i<=N);
        nodes_[i] = {}; nodes_[i].next = free_; free_ = i; ++available_;
    }
    std::size_t available() const noexcept { return available_; }
    Order& operator[](Index i) noexcept { assert(i>0 && i<=N); return nodes_[i]; }
    const Order& operator[](Index i) const noexcept { assert(i>0 && i<=N); return nodes_[i]; }
    bool check_free_list() const noexcept {
        std::size_t n=0;
        for (Index i=free_; i; i=nodes_[i].next) {
            if (i>N || ++n>N || nodes_[i].id) return false;
        }
        return n==available_;
    }
};

// 5. SPSC only. Publication of payload precedes release of write_; acquire on
// consumer observes payload. Read cursor is published AFTER payload is copied.
// Producer-local and consumer-local cached cursors reduce coherence traffic.
// Monotonic uint64 counters wrap modulo 2^64; occupancy remains <= Capacity.
template<class T, std::size_t Capacity>
class SpscRing {
    static_assert(power_of_two(Capacity) && Capacity < (1ULL<<63));
    static_assert(std::is_trivially_copyable_v<T>);
    static_assert(std::atomic<std::uint64_t>::is_always_lock_free);
    alignas(CacheLine) T data_[Capacity]{};
    alignas(CacheLine) std::atomic<std::uint64_t> write_{0};
    alignas(CacheLine) std::atomic<std::uint64_t> read_{0};
    alignas(CacheLine) std::uint64_t cached_read_ = 0;  // producer only
    alignas(CacheLine) std::uint64_t cached_write_ = 0; // consumer only
public:
    SpscRing() = default;
    SpscRing(const SpscRing&) = delete;
    SpscRing& operator=(const SpscRing&) = delete;
    static constexpr std::size_t capacity() noexcept { return Capacity; }
    // Must be called by producer. The consumer can only increase free space.
    std::size_t producer_free() noexcept {
        auto w = write_.load(std::memory_order_relaxed);
        cached_read_ = read_.load(std::memory_order_acquire);
        return Capacity-std::size_t(w-cached_read_);
    }
    bool try_push(const T& v) noexcept {
        auto w = write_.load(std::memory_order_relaxed);
        if (w-cached_read_ == Capacity) {
            cached_read_ = read_.load(std::memory_order_acquire);
            if (w-cached_read_ == Capacity) return false;
        }
        data_[w & (Capacity-1)] = v;
        write_.store(w+1, std::memory_order_release);
        return true;
    }
    bool try_pop(T& out) noexcept {
        auto r = read_.load(std::memory_order_relaxed);
        if (r == cached_write_) {
            cached_write_ = write_.load(std::memory_order_acquire);
            if (r == cached_write_) return false;
        }
        out = data_[r & (Capacity-1)];
        read_.store(r+1, std::memory_order_release);
        return true;
    }
};

// 6. Flat ladders + pool + ID map + intrusive per-level FIFO + bitmap summaries.
template<Tick BaseTick, std::size_t PriceCount, std::size_t MaxOrders,
         std::size_t HashCapacity>
class OrderBook {
    static_assert(PriceCount>0 && PriceCount<=std::size_t(std::numeric_limits<int>::max()));
    static_assert(std::int64_t(BaseTick)+std::int64_t(PriceCount)-1 <= std::numeric_limits<Tick>::max());
    static_assert(HashCapacity/2 >= MaxOrders, "Keep hash load <= 50%");
    alignas(CacheLine) PriceLevel levels_[2][PriceCount]{};
    Bitmap<PriceCount> occupied_[2];
    OrderPool<MaxOrders> pool_;
    IdHash<HashCapacity> ids_;
    int best_[2] = {-1,-1};
    std::uint64_t sequence_ = 0;
    static int sidx(Side side) noexcept { return side==Side::Buy ? 0 : 1; }
    static Tick price(int offset) noexcept { return Tick(std::int64_t(BaseTick)+offset); }
    static bool valid_price(Tick p) noexcept {
        auto off = std::int64_t(p)-BaseTick;
        return off>=0 && off<std::int64_t(PriceCount);
    }
    static int offset(Tick p) noexcept { return int(std::int64_t(p)-BaseTick); }
    int next_price(int side, int current) const noexcept {
        return side==0 ? occupied_[0].prev(current-1) : occupied_[1].next(current+1);
    }
    bool crosses(const Command& c, int level) const noexcept {
        return level>=0 && (c.type==Type::Market ||
               (c.side==Side::Buy ? price(level)<=c.tick : price(level)>=c.tick));
    }
    // Calculate executable qty using aggregated depth; also whether at least
    // one maker would be removed (needed only for pool-full GTC admission).
    struct Preview { Qty leaves; bool frees_slot; };
    Preview preview(const Command& c) const noexcept {
        const int opposite = 1-sidx(c.side);
        Qty rem = c.qty; bool frees = false;
        for (int p=best_[opposite]; rem && crosses(c,p); p=next_price(opposite,p)) {
            const auto& l = levels_[opposite][p];
            if (rem >= pool_[l.head].qty) frees = true;
            if (l.total_qty >= rem) return {0,frees};
            rem -= Qty(l.total_qty); // total < rem <= UINT32_MAX here
        }
        return {rem,frees};
    }
    void unlink(Index i) noexcept {
        auto& o=pool_[i]; const int s=sidx(o.side); const int p=int(o.level);
        auto& l=levels_[s][p];
        if (o.prev) pool_[o.prev].next=o.next; else l.head=o.next;
        if (o.next) pool_[o.next].prev=o.prev; else l.tail=o.prev;
        l.total_qty-=o.qty; --l.count;
        o.prev=o.next=0;
        if (!l.count) {
            occupied_[s].clear(p);
            if (best_[s]==p) best_[s]=next_price(s,p);
        }
    }
    void remove(Index i) noexcept {
        const Id id=pool_[i].id;
        unlink(i); require_internal(ids_.erase(id)); pool_.release(i);
    }
    void append(Index i, const Command& c, Qty qty) noexcept {
        auto& o=pool_[i]; o={};
        o.id=c.id; o.qty=qty; o.level=std::uint32_t(offset(c.tick)); o.side=c.side;
        const int s=sidx(c.side); auto& l=levels_[s][o.level];
        o.prev=l.tail;
        if (l.tail) pool_[l.tail].next=i; else l.head=i;
        l.tail=i; l.total_qty+=qty; ++l.count;
        require_internal(ids_.insert(o.id,i));
        if (l.count==1) {
            occupied_[s].set(int(o.level));
            if (best_[s]<0 || (s==0 ? int(o.level)>best_[s] : int(o.level)<best_[s]))
                best_[s]=int(o.level);
        }
    }
public:
    static constexpr std::size_t max_orders=MaxOrders;
    static constexpr std::size_t max_events=MaxOrders+1;
    using EventBatch=FixedVector<Event,max_events>;
    OrderBook() = default;
    OrderBook(const OrderBook&) = delete;
    OrderBook& operator=(const OrderBook&) = delete;
private:
    void emit(EventBatch& out, Event e) noexcept {
        e.sequence=sequence_; require_internal(out.push_back(e));
    }
    void reject(const Command& c, Reason why, EventBatch& out) noexcept {
        emit(out,{0,c.id,0,c.tick,0,0,0,c.side,EventKind::Rejected,why});
    }
    void done(const Command& c, Qty filled, Qty leaves, Qty canceled, EventBatch& out) noexcept {
        emit(out,{0,c.id,0,c.tick,filled,leaves,canceled,c.side,EventKind::Done,Reason::None});
    }
    // reserved is either zero or an already detached replacement node.
    void execute(const Command& c, EventBatch& out, Index reserved=0) noexcept {
        Qty rem=c.qty; const int opposite=1-sidx(c.side);
        while (rem && crosses(c,best_[opposite])) {
            const int p=best_[opposite]; auto& l=levels_[opposite][p];
            const Index i=l.head; auto& maker=pool_[i];
            const Qty traded=smaller(rem,maker.qty);
            const Id maker_id=maker.id;
            rem-=traded; maker.qty-=traded; l.total_qty-=traded;
            // unlink subtracts the remaining qty (zero when fully filled).
            if (!maker.qty) remove(i);
            emit(out,{0,c.id,maker_id,price(p),traded,0,0,c.side,EventKind::Trade,Reason::None});
        }
        const bool rest=rem && (c.tif==Tif::GTC || c.tif==Tif::PostOnly);
        if (rest) {
            const Index i=reserved ? reserved : pool_.acquire();
            require_internal(i!=0); append(i,c,rem); reserved=0;
        }
        if (reserved) pool_.release(reserved);
        done(c,c.qty-rem,rest ? rem : 0,rest ? 0 : rem,out);
    }
    void add(const Command& c, EventBatch& out) noexcept {
        if (!c.id) return reject(c,Reason::BadId,out);
        if (c.side!=Side::Buy && c.side!=Side::Sell) return reject(c,Reason::BadSide,out);
        if (c.type!=Type::Limit && c.type!=Type::Market) return reject(c,Reason::BadType,out);
        if (c.tif!=Tif::GTC && c.tif!=Tif::IOC && c.tif!=Tif::FOK && c.tif!=Tif::PostOnly)
            return reject(c,Reason::BadTif,out);
        if (c.type==Type::Market && c.tif!=Tif::IOC && c.tif!=Tif::FOK)
            return reject(c,Reason::BadTif,out);
        if (!c.qty) return reject(c,Reason::BadQty,out);
        if (c.type==Type::Limit && !valid_price(c.tick)) return reject(c,Reason::BadPrice,out);
        if (ids_.find(c.id)) return reject(c,Reason::DuplicateId,out);
        if (c.tif==Tif::PostOnly && crosses(c,best_[1-sidx(c.side)]))
            return reject(c,Reason::WouldCross,out);
        if (c.tif==Tif::FOK) {
            if (preview(c).leaves) return reject(c,Reason::FokUnfilled,out);
        } else if ((c.tif==Tif::GTC || c.tif==Tif::PostOnly) && !pool_.available()) {
            const auto p=preview(c);
            if (p.leaves && !p.frees_slot) return reject(c,Reason::PoolFull,out);
        }
        execute(c,out);
    }
    void cancel_or_replace(const Command& input, EventBatch& out) noexcept {
        if (!input.id) return reject(input,Reason::BadId,out);
        const Index i=ids_.find(input.id);
        if (!i) return reject(input,Reason::UnknownId,out);
        auto& old=pool_[i];
        Command c=input; c.side=old.side; // these commands identify side by ID
        if (c.action==Action::Cancel || !c.qty) {
            const Qty removed=old.qty;
            remove(i); return done(c,0,0,removed,out);
        }
        if (!valid_price(c.tick)) return reject(input,Reason::BadPrice,out);
        if (offset(c.tick)==int(old.level) && c.qty<=old.qty) {
            const Qty reduction=old.qty-c.qty;
            levels_[sidx(old.side)][old.level].total_qty-=reduction;
            old.qty=c.qty; return done(c,0,c.qty,reduction,out);
        }
        // Validate first, then detach and retain the slot. Nothing fallible
        // remains after detaching; no cancel-then-reject loss of the old order.
        unlink(i); require_internal(ids_.erase(c.id));
        c.type=Type::Limit; c.tif=Tif::GTC;
        execute(c,out,i);
    }
public:
    void process(const Command& c, EventBatch& out) noexcept {
        out.clear(); ++sequence_;
        switch(c.action) {
            case Action::Add: add(c,out); break;
            case Action::Cancel: case Action::Replace: cancel_or_replace(c,out); break;
            default: reject(c,Reason::BadAction,out); break;
        }
    }
    std::size_t size() const noexcept { return ids_.size(); }
    std::size_t free_slots() const noexcept { return pool_.available(); }
    bool find(Id id, OrderView& out) const noexcept {
        const Index i=ids_.find(id); if (!i) return false;
        const auto& o=pool_[i]; out={o.id,o.qty,price(int(o.level)),o.side}; return true;
    }
    bool best(Side side, LevelView& out) const noexcept {
        if (side!=Side::Buy && side!=Side::Sell) return false;
        const int s=sidx(side), p=best_[s]; if (p<0) return false;
        const auto& l=levels_[s][p]; out={price(p),l.total_qty,l.count}; return true;
    }
    bool level(Side side, Tick tick, LevelView& out) const noexcept {
        if ((side!=Side::Buy && side!=Side::Sell) || !valid_price(tick)) return false;
        const auto& l=levels_[sidx(side)][offset(tick)];
        out={tick,l.total_qty,l.count}; return true;
    }
    // Price-ordered traversal (best to worst). Callback returns false to stop.
    // Inspection only: callback must not mutate this book or re-enter process.
    template<class Fn> void for_each_level(Side side, Fn&& fn) const {
        if (side!=Side::Buy && side!=Side::Sell) return;
        const int s=sidx(side);
        for (int p=best_[s]; p>=0; p=next_price(s,p)) {
            const auto& l=levels_[s][p];
            if (!fn(LevelView{price(p),l.total_qty,l.count})) break;
        }
    }
    template<class Fn> void for_each_order(Side side, Tick tick, Fn&& fn) const {
        if ((side!=Side::Buy && side!=Side::Sell) || !valid_price(tick)) return;
        for (Index i=levels_[sidx(side)][offset(tick)].head; i; i=pool_[i].next) {
            const auto& o=pool_[i];
            if (!fn(OrderView{o.id,o.qty,tick,o.side})) break;
        }
    }
    // Cold-path structural audit. Independent of asserts and enabled in release.
    bool validate() const noexcept {
        if (!pool_.check_free_list()) return false;
        bool seen[MaxOrders+1]{}; std::size_t count=0;
        for (int s=0;s<2;++s) {
            const int b=s==0 ? occupied_[s].prev(int(PriceCount)-1) : occupied_[s].next(0);
            if (b!=best_[s]) return false;
            for (std::size_t p=0;p<PriceCount;++p) {
                const auto& l=levels_[s][p];
                if (occupied_[s].test(int(p)) != (l.count!=0)) return false;
                std::uint64_t sum=0; std::uint32_t n=0; Index prev=0;
                for (Index i=l.head;i;i=pool_[i].next) {
                    if (i>MaxOrders || seen[i] || ++n>MaxOrders) return false;
                    seen[i]=true; const auto& o=pool_[i];
                    if (!o.id || !o.qty || o.prev!=prev || o.level!=p || sidx(o.side)!=s || ids_.find(o.id)!=i)
                        return false;
                    sum+=o.qty; prev=i;
                }
                if (n!=l.count || sum!=l.total_qty || prev!=l.tail) return false;
                count+=n;
            }
        }
        return count==ids_.size() && count+pool_.available()==MaxOrders &&
               !(best_[0]>=0 && best_[1]>=0 && best_[0]>=best_[1]);
    }
};

// 7. One input producer -> matching owner -> one event consumer. pump_one is
// called ONLY by matching owner. Client retains/retries a command on false
// try_submit. A false pump_one means no input or output backpressure; never drop.
// Consumers can group events by sequence and the terminal Done/Rejected marker.
template<class Book, std::size_t InputCapacity, std::size_t OutputCapacity>
class QueuedEngine {
    static_assert(OutputCapacity>=Book::max_events);
    Book book_;
    SpscRing<Command,InputCapacity> input_;
    SpscRing<Event,OutputCapacity> output_;
    typename Book::EventBatch batch_;
public:
    bool try_submit(const Command& c) noexcept { return input_.try_push(c); }
    bool try_event(Event& e) noexcept { return output_.try_pop(e); }
    bool pump_one() noexcept {
        if (output_.producer_free()<Book::max_events) return false;
        Command c;
        if (!input_.try_pop(c)) return false;
        book_.process(c,batch_);
        for (const auto& e:batch_) require_internal(output_.try_push(e));
        return true;
    }
};
} // namespace hft

#ifndef HFT_NO_MAIN
namespace app {
using namespace hft;
#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#expr); std::abort(); \
} } while(false)
Command limit(Id id, Side side, Tick p, Qty q, Tif tif=Tif::GTC) {
    return {Action::Add,side,Type::Limit,tif,id,p,q};
}
Command market(Id id, Side side, Qty q, Tif tif=Tif::IOC) {
    return {Action::Add,side,Type::Market,tif,id,0,q};
}
Command cancel(Id id) { return {Action::Cancel,Side::Buy,Type::Limit,Tif::GTC,id,0,0}; }
Command replace(Id id, Tick p, Qty q) { return {Action::Replace,Side::Buy,Type::Limit,Tif::GTC,id,p,q}; }
struct Rng {
    std::uint64_t state;
    std::uint64_t next() noexcept {
        state ^= state>>12; state ^= state<<25; state ^= state>>27;
        return state*2685821657736338717ULL;
    }
};

// 8. Slow oracle: unsorted array, linear ID lookup, scan EVERY live order for
// best price then arrival serial. No production hash, bitmap, pool or links.
template<Tick Base, std::size_t Prices, std::size_t Max>
class Reference {
    struct Node { OrderView o{}; std::uint64_t arrival=0; };
    Node orders_[Max]{};
    std::uint64_t arrival_=0, sequence_=0;
    int locate(Id id) const {
        if (!id) return -1;
        for (std::size_t i=0;i<Max;++i) if (orders_[i].o.id==id) return int(i);
        return -1;
    }
    bool valid(Tick p) const { return std::int64_t(p)>=Base && std::int64_t(p)<std::int64_t(Base)+std::int64_t(Prices); }
    bool eligible(const Command& c, const OrderView& o) const {
        return o.id && o.side!=c.side && (c.type==Type::Market ||
               (c.side==Side::Buy ? o.tick<=c.tick : o.tick>=c.tick));
    }
    int maker(const Command& c) const {
        int b=-1;
        for (std::size_t i=0;i<Max;++i) {
            const auto& n=orders_[i]; if (!eligible(c,n.o)) continue;
            if (b<0) { b=int(i); continue; }
            const auto& old=orders_[b];
            bool better=c.side==Side::Buy ? n.o.tick<old.o.tick : n.o.tick>old.o.tick;
            if (better || (n.o.tick==old.o.tick && n.arrival<old.arrival)) b=int(i);
        }
        return b;
    }
public:
    using Batch=FixedVector<Event,Max+1>;
    std::size_t size() const { std::size_t n=0; for (const auto& x:orders_) n+=x.o.id!=0; return n; }
    bool find(Id id, OrderView& out) const { int i=locate(id); if(i<0)return false; out=orders_[i].o; return true; }
    LevelView level(Side side, Tick p) const {
        LevelView v{p,0,0}; for(const auto& x:orders_) if(x.o.id && x.o.side==side && x.o.tick==p) {
            v.qty+=x.o.qty; ++v.count;
        } return v;
    }
    void process(Command c, Batch& out) {
        out.clear(); ++sequence_;
        auto emit=[&](Event e) { e.sequence=sequence_; CHECK(out.push_back(e)); };
        auto reject=[&](Reason r) { emit({0,c.id,0,c.tick,0,0,0,c.side,EventKind::Rejected,r}); };
        auto done=[&](Qty q,Qty l,Qty x) { emit({0,c.id,0,c.tick,q,l,x,c.side,EventKind::Done,Reason::None}); };
        if(c.action!=Action::Add && c.action!=Action::Cancel && c.action!=Action::Replace) return reject(Reason::BadAction);
        if(!c.id) return reject(Reason::BadId);
        if(c.action!=Action::Add) {
            int i=locate(c.id); if(i<0)return reject(Reason::UnknownId);
            auto old=orders_[i].o;
            if(c.action==Action::Cancel || !c.qty) {
                c.side=old.side; orders_[i]={}; return done(0,0,old.qty);
            }
            if(!valid(c.tick)) return reject(Reason::BadPrice);
            c.side=old.side;
            if(c.tick==old.tick && c.qty<=old.qty) {
                orders_[i].o.qty=c.qty; return done(0,c.qty,old.qty-c.qty);
            }
            orders_[i]={}; c.type=Type::Limit; c.tif=Tif::GTC;
        } else {
            if(c.side!=Side::Buy && c.side!=Side::Sell) return reject(Reason::BadSide);
            if(c.type!=Type::Limit && c.type!=Type::Market) return reject(Reason::BadType);
            if(c.tif!=Tif::GTC && c.tif!=Tif::IOC && c.tif!=Tif::FOK && c.tif!=Tif::PostOnly) return reject(Reason::BadTif);
            if(c.type==Type::Market && c.tif!=Tif::IOC && c.tif!=Tif::FOK) return reject(Reason::BadTif);
            if(!c.qty) return reject(Reason::BadQty);
            if(c.type==Type::Limit && !valid(c.tick)) return reject(Reason::BadPrice);
            if(locate(c.id)>=0) return reject(Reason::DuplicateId);
            if(c.tif==Tif::PostOnly && maker(c)>=0) return reject(Reason::WouldCross);
            std::uint64_t available=0;
            for(const auto& n:orders_) if(eligible(c,n.o)) available+=n.o.qty;
            if(c.tif==Tif::FOK && available<c.qty) return reject(Reason::FokUnfilled);
            // A residual implies EVERY eligible maker is consumed, thus if
            // any eligible maker exists a pool slot will have become free.
            if((c.tif==Tif::GTC || c.tif==Tif::PostOnly) && size()==Max &&
               available<c.qty && !available) return reject(Reason::PoolFull);
        }
        Qty rem=c.qty;
        while(rem) {
            int i=maker(c); if(i<0)break;
            auto& o=orders_[i].o;
            Qty q=smaller(rem,o.qty);
            emit({0,c.id,o.id,o.tick,q,0,0,c.side,EventKind::Trade,Reason::None});
            rem-=q; o.qty-=q; if(!o.qty)orders_[i]={};
        }
        bool rest=rem && (c.tif==Tif::GTC || c.tif==Tif::PostOnly);
        if(rest) {
            int i=-1; for(std::size_t j=0;j<Max;++j) if(!orders_[j].o.id){i=int(j);break;}
            CHECK(i>=0); orders_[i]={{c.id,rem,c.tick,c.side},++arrival_};
        }
        done(c.qty-rem,rest?rem:0,rest?0:rem);
    }
};

void container_tests() {
    FixedVector<int,2> v; CHECK(v.push_back(1)); CHECK(v.push_back(2));
    CHECK(!v.push_back(3)); CHECK(v.size()==2 && v[1]==2); v.clear(); CHECK(v.empty());
    // Full hash misses, wrapped clusters, erase/reinsert, duplicate keys.
    IdHash<8> h; Id collision[8]{}; int n=0;
    for(Id id=1;n<8;++id) if((IdHash<8>::hash(id)&7)==7) collision[n++]=id;
    for(int i=0;i<8;++i) CHECK(h.insert(collision[i],Index(i+1)));
    CHECK(!h.find(999999)); CHECK(!h.insert(999999,9)); CHECK(!h.erase(999999));
    CHECK(!h.insert(collision[3],8)); CHECK(!h.insert(0,1)); CHECK(!h.erase(0));
    for(int i=0;i<8;++i) {
        CHECK(h.erase(collision[i])); CHECK(!h.find(collision[i]));
        for(int j=i+1;j<8;++j) CHECK(h.find(collision[j])==Index(j+1));
    }
    Rng rng{0x123456}; IdHash<64> hash; bool present[101]{};
    for(int k=0;k<30000;++k) {
        Id id=1+rng.next()%100;
        if(rng.next()&1) {
            bool expected=!present[id] && hash.size()<64;
            CHECK(hash.insert(id,Index(id))==expected); if(expected)present[id]=true;
        } else { CHECK(hash.erase(id)==present[id]); present[id]=false; }
        for(Id j=1;j<=100;++j) CHECK(hash.find(j)==(present[j]?j:0));
    }
    // Three-level hierarchy and a partial final word.
    Bitmap<5003> bits; bool model[5003]{};
    CHECK(bits.next(0)==-1 && bits.prev(9000)==-1);
    for(int k=0;k<15000;++k) {
        int p=int(rng.next()%5003); bool set=(rng.next()&1)!=0;
        if(set)bits.set(p);else bits.clear(p); model[p]=set;
        int from=int(rng.next()%5100)-40;
        int a=from<0?0:from; while(a<5003 && !model[a])++a;
        int b=from>=5003?5002:from; while(b>=0 && !model[b])--b;
        CHECK(bits.next(from)==(a>=5003?-1:a)); CHECK(bits.prev(from)==(b<0?-1:b));
    }
    for(int p=0;p<5003;++p)bits.clear(p);
    CHECK(bits.next(0)==-1 && bits.prev(5002)==-1);
    Bitmap<1> one; one.set(0); CHECK(one.next(-10)==0 && one.prev(100)==0); one.clear(0);
    OrderPool<2> pool; Index a=pool.acquire(),b=pool.acquire(); CHECK(a && b && a!=b);
    CHECK(pool.acquire()==0); pool.release(a); CHECK(pool.acquire()==a);
    pool.release(a); pool.release(b); CHECK(pool.check_free_list());
}

void deterministic_tests() {
    using B=OrderBook<90,64,8,16>;
    B book; B::EventBatch out; OrderView view; LevelView lv;
    auto run=[&](Command c) {book.process(c,out);CHECK(book.validate());};
    run(limit(1,Side::Sell,100,5)); run(limit(2,Side::Sell,100,7));
    run(limit(3,Side::Sell,101,10));
    run(limit(4,Side::Buy,101,8));
    CHECK(out.size()==3 && out[0].maker_id==1 && out[0].qty==5);
    CHECK(out[1].maker_id==2 && out[1].qty==3 && out[1].tick==100);
    CHECK(book.find(2,view) && view.qty==4);
    run(limit(5,Side::Buy,101,20,Tif::FOK)); CHECK(out[0].reason==Reason::FokUnfilled);
    CHECK(book.find(2,view) && view.qty==4);
    run(limit(6,Side::Buy,100,1,Tif::PostOnly));CHECK(out[0].reason==Reason::WouldCross);
    run(replace(2,100,2)); CHECK(out[0].canceled==2);
    run(limit(7,Side::Sell,100,3));
    run(replace(2,100,5)); // quantity increase goes behind ID 7
    run(market(8,Side::Buy,4));
    CHECK(out[0].maker_id==7 && out[1].maker_id==2);
    run(replace(2,1000,6));CHECK(out[0].reason==Reason::BadPrice);
    CHECK(book.find(2,view) && view.qty==4);
    run(replace(2,100,0)); CHECK(!book.find(2,view));
    run(market(9,Side::Buy,100));CHECK(out[out.size()-1].canceled==90);
    CHECK(!book.best(Side::Sell,lv));
    run(limit(10,Side::Buy,99,10));run(limit(10,Side::Sell,100,10));
    CHECK(out[0].reason==Reason::DuplicateId);
    run(limit(11,Side::Sell,98,3)); CHECK(out[0].tick==99);
    run(cancel(10));CHECK(book.size()==0);
    run(market(12,Side::Sell,5,Tif::FOK));CHECK(out[0].reason==Reason::FokUnfilled);
    run(limit(13,Side::Buy,90,5));run(limit(14,Side::Sell,153,5));
    CHECK(book.best(Side::Buy,lv) && lv.tick==90);
    CHECK(book.best(Side::Sell,lv) && lv.tick==153);
    // Same-price reduction must retain priority, unlike quantity increase.
    B fifo; fifo.process(limit(1,Side::Sell,100,5),out);
    fifo.process(limit(2,Side::Sell,100,5),out);
    fifo.process(replace(1,100,3),out);fifo.process(market(3,Side::Buy,4),out);
    CHECK(out[0].maker_id==1 && out[0].qty==3 && out[1].maker_id==2);
    CHECK(fifo.validate());
    // Aggregate volume must exceed 32 bits without wrapping.
    B wide; wide.process(limit(1,Side::Buy,90,UINT32_MAX),out);
    wide.process(limit(2,Side::Buy,90,UINT32_MAX),out);
    CHECK(wide.best(Side::Buy,lv) && lv.qty==2ULL*UINT32_MAX && wide.validate());
    // Full pool: rejection is atomic; replace and executable adds still work.
    using Tiny=OrderBook<0,4,2,4>; Tiny full; Tiny::EventBatch events;
    full.process(limit(1,Side::Buy,1,2),events);full.process(limit(2,Side::Sell,3,2),events);
    full.process(limit(3,Side::Buy,2,4),events);CHECK(events[0].reason==Reason::PoolFull);
    CHECK(full.find(1,view) && view.qty==2);
    full.process(replace(1,2,5),events);CHECK(events[0].kind==EventKind::Done && full.validate());
    full.process(limit(3,Side::Buy,3,1),events);CHECK(events[0].kind==EventKind::Trade && full.size()==2);
    full.process(limit(4,Side::Buy,3,5),events);CHECK(events[0].qty==1 && full.find(4,view) && view.qty==4);
    CHECK(full.validate());
    // Price-changing replace can cross and execute at the maker's tick.
    B repr; repr.process(limit(1,Side::Buy,99,3),out);
    repr.process(limit(2,Side::Sell,101,3),out);repr.process(replace(1,102,5),out);
    CHECK(out[0].tick==101 && out[0].maker_id==2 && repr.find(1,view) && view.qty==2 && view.tick==102);
    CHECK(repr.validate());
    // Sweep maximum number of makers: exact output bound N+1.
    B sweep;for(Id id=1;id<=8;++id)sweep.process(limit(id,Side::Sell,100,1),out);
    sweep.process(market(9,Side::Buy,8,Tif::FOK),out);
    CHECK(out.size()==B::max_events && sweep.size()==0 && sweep.validate());
    // Negative and highest representable tick bases.
    OrderBook<-100,4,2,4> neg; decltype(neg)::EventBatch nb;
    neg.process(limit(1,Side::Buy,-100,1),nb);CHECK(neg.best(Side::Buy,lv) && lv.tick==-100);
    OrderBook<INT32_MAX-1,2,2,4> high; decltype(high)::EventBatch hb;
    high.process(limit(1,Side::Sell,INT32_MAX,1),hb);CHECK(high.best(Side::Sell,lv) && lv.tick==INT32_MAX);
}

void randomized_tests() {
    constexpr int Prices=130; constexpr std::size_t Max=32;
    using B=OrderBook<-20,Prices,Max,64>;
    // 12 independent streams, 300,000 commands. Compare every event, every
    // live ID and every level after each command; audit internal links too.
    for(std::uint64_t seed=1;seed<=12;++seed) {
        B book; Reference<-20,Prices,Max> ref; B::EventBatch got,want;
        Rng rng{seed*0x9e3779b97f4a7c15ULL};
        for(int step=0;step<25000;++step) {
            Command c=limit(1+rng.next()%64,Side(rng.next()%2),Tick(int(rng.next()%(Prices+4))-22),Qty(1+rng.next()%30),Tif(rng.next()%4));
            auto op=rng.next()%10;
            if(op<2)c.action=Action::Cancel;
            else if(op<4){c.action=Action::Replace;c.qty=Qty(rng.next()%35);}
            else if(op==4){c.type=Type::Market;c.tif=(rng.next()&1)?Tif::IOC:Tif::FOK;}
            // Inject invalid enum values and boundary data through the API.
            switch(rng.next()%101) {
                case 0:c.id=0;break; case 1:c.qty=0;break;
                case 2:c.side=Side(9);break;case 3:c.type=Type(9);break;
                case 4:c.tif=Tif(9);break;case 5:c.action=Action(9);break;
                case 6:c.qty=UINT32_MAX;break;default:break;
            }
            book.process(c,got);ref.process(c,want);
            if(got.size()!=want.size()) {
                std::fprintf(stderr,"event count seed=%llu step=%d\n",(unsigned long long)seed,step); CHECK(false);
            }
            for(std::size_t i=0;i<got.size();++i) if(!(got[i]==want[i])) {
                std::fprintf(stderr,"event mismatch seed=%llu step=%d event=%zu\n",(unsigned long long)seed,step,i); CHECK(false);
            }
            CHECK(book.validate() && book.size()==ref.size());
            for(Id id=1;id<=64;++id) {
                OrderView a,b;bool x=book.find(id,a),y=ref.find(id,b);
                CHECK(x==y);if(x)CHECK(a==b);
            }
            for(int s=0;s<2;++s)for(int p=-20;p<-20+Prices;++p) {
                LevelView a;CHECK(book.level(Side(s),p,a)); CHECK(a==ref.level(Side(s),p));
            }
        }
    }
}

void queue_tests() {
    SpscRing<std::uint64_t,4> small;std::uint64_t value=0;
    CHECK(!small.try_pop(value));
    for(std::uint64_t i=0;i<4;++i)CHECK(small.try_push(i));
    CHECK(!small.try_push(9) && small.producer_free()==0);
    for(std::uint64_t i=0;i<4;++i){CHECK(small.try_pop(value));CHECK(value==i);}
    CHECK(!small.try_pop(value));
    SpscRing<std::uint64_t,1024> ring;
    constexpr std::uint64_t N=1000000;
    std::thread producer([&] {for(std::uint64_t i=0;i<N;++i)while(!ring.try_push(i))std::this_thread::yield();});
    for(std::uint64_t i=0;i<N;++i) {
        while(!ring.try_pop(value)) std::this_thread::yield();
        CHECK(value==i);
    }
    producer.join(); CHECK(!ring.try_pop(value));
    // Deliberately stop draining output: pump stalls without consuming input.
    using B=OrderBook<0,8,2,4>;
    QueuedEngine<B,4,4> pipeline;Event e;
    CHECK(pipeline.try_submit(limit(1,Side::Buy,1,1)));
    CHECK(pipeline.try_submit(limit(2,Side::Buy,1,1)));
    CHECK(pipeline.try_submit(cancel(1)));
    CHECK(pipeline.pump_one());CHECK(pipeline.pump_one());CHECK(!pipeline.pump_one());
    CHECK(pipeline.try_event(e) && e.id==1);
    CHECK(pipeline.pump_one());CHECK(pipeline.try_event(e) && e.id==2);
    CHECK(pipeline.try_event(e) && e.id==1 && e.canceled==1);CHECK(!pipeline.try_event(e));
}

const char* reason_name(Reason r) {
    switch(r) {
        case Reason::None:return "None";case Reason::BadAction:return "BadAction";
        case Reason::BadId:return "BadId";case Reason::BadSide:return "BadSide";
        case Reason::BadType:return "BadType";case Reason::BadTif:return "BadTif";
        case Reason::BadQty:return "BadQty";case Reason::BadPrice:return "BadPrice";
        case Reason::DuplicateId:return "DuplicateId";case Reason::UnknownId:return "UnknownId";
        case Reason::PoolFull:return "PoolFull";case Reason::WouldCross:return "WouldCross";
        case Reason::FokUnfilled:return "FokUnfilled";
    }return "InvalidReason";
}
void print_event(const Event& e) {
    if(e.kind==EventKind::Trade)
        std::printf("seq=%llu TRADE maker=%llu taker=%llu side=%c tick=%d qty=%u\n",
            (unsigned long long)e.sequence,(unsigned long long)e.maker_id,(unsigned long long)e.id,
            e.side==Side::Buy?'B':'S',e.tick,e.qty);
    else if(e.kind==EventKind::Rejected)
        std::printf("seq=%llu REJECT id=%llu reason=%s\n",(unsigned long long)e.sequence,(unsigned long long)e.id,reason_name(e.reason));
    else
        std::printf("seq=%llu DONE id=%llu filled=%u leaves=%u canceled=%u\n",
            (unsigned long long)e.sequence,(unsigned long long)e.id,e.qty,e.leaves,e.canceled);
}
using DemoBook=OrderBook<0,65536,8192,16384>;
void print_depth(const DemoBook& book) {
    const Side sides[]={Side::Buy,Side::Sell};
    for(Side side:sides) {
        std::printf("%s (best first):\n",side==Side::Buy?"BIDS":"ASKS");
        int n=0;book.for_each_level(side,[&](LevelView v) {
            std::printf("  tick=%d qty=%llu orders=%u\n",v.tick,(unsigned long long)v.qty,v.count);
            return ++n<10;
        });
    }
}
void demo() {
    static DemoBook book;static DemoBook::EventBatch events;
    const Command commands[]={limit(1,Side::Sell,100,5),limit(2,Side::Sell,100,7),
        limit(3,Side::Sell,101,8),limit(4,Side::Buy,101,10),replace(2,100,1),
        limit(5,Side::Buy,101,50,Tif::FOK),market(6,Side::Buy,4),
        limit(7,Side::Buy,99,10),cancel(7)};
    for(const auto& c:commands){book.process(c,events);for(const auto& e:events)print_event(e);}
    print_depth(book);CHECK(book.validate());
}

// CLI is cold-path text I/O, not a production wire decoder. Strict conversion
// rejects overflow/trailing junk, avoiding scanf integer-overflow behavior.
struct Tokens { char* v[8]{};int n=0;bool overflow=false; };
Tokens tokenize(char* line) {
    Tokens t;char* p=line;
    while(*p) {
        while(*p==' '||*p=='\t'||*p=='\r'||*p=='\n')++p;
        if(!*p || *p=='#')break;
        if(t.n==8){t.overflow=true;break;}t.v[t.n++]=p;
        while(*p && *p!=' ' && *p!='\t' && *p!='\r' && *p!='\n')++p;
        if(*p)*p++='\0';
    }return t;
}
bool unsigned_number(const char* s, std::uint64_t max, std::uint64_t& out) {
    if(!s || !*s)return false;
    std::uint64_t n=0;
    for(;*s;++s) {if(*s<'0'||*s>'9')return false;unsigned d=unsigned(*s-'0');
        if(n>max/10 || (n==max/10 && d>max%10))return false;
        n=n*10+d;
    }
    out=n;return true;
}
bool tick_number(const char* s,Tick& out) {
    bool negative=*s=='-';if(negative)++s;
    std::uint64_t n=0;if(!unsigned_number(s,negative?2147483648ULL:2147483647ULL,n))return false;
    out=Tick(negative?-std::int64_t(n):std::int64_t(n));return true;
}
bool eq(const char* a,const char* b){return std::strcmp(a,b)==0;}
bool parse_tif(const char* s,Tif& t) {
    if(eq(s,"GTC"))t=Tif::GTC;else if(eq(s,"IOC"))t=Tif::IOC;
    else if(eq(s,"FOK"))t=Tif::FOK;else if(eq(s,"POST"))t=Tif::PostOnly;else return false;
    return true;
}
bool parse_command(const Tokens& t,Command& c) {
    if(t.overflow || !t.n)return false;
    std::uint64_t id=0,q=0;Tick p=0;
    if(eq(t.v[0],"C") && t.n==2) {
        if(!unsigned_number(t.v[1],UINT64_MAX,id))return false;
        c=cancel(id);return true;
    }
    if(eq(t.v[0],"R") && t.n==4) {
        if(!unsigned_number(t.v[1],UINT64_MAX,id)||!tick_number(t.v[2],p)||!unsigned_number(t.v[3],UINT32_MAX,q))return false;
        c=replace(id,p,Qty(q));return true;
    }
    bool m=eq(t.v[0],"M"),l=eq(t.v[0],"L");
    if((!m&&!l)||(m?t.n!=5:t.n!=6))return false;
    if(!unsigned_number(t.v[1],UINT64_MAX,id))return false;
    Side side;if(eq(t.v[2],"B"))side=Side::Buy;else if(eq(t.v[2],"S"))side=Side::Sell;else return false;
    if(l&&!tick_number(t.v[3],p))return false;
    if(!unsigned_number(t.v[m?3:4],UINT32_MAX,q))return false;
    Tif tif;if(!parse_tif(t.v[m?4:5],tif))return false;
    c=m?market(id,side,Qty(q),tif):limit(id,side,p,Qty(q),tif);return true;
}
void cli() {
    static DemoBook book;static DemoBook::EventBatch events;
    std::puts("Ticks 0..65535; max 8192 live orders. R quantity = new REMAINING qty.\n"
              "L id B|S tick qty GTC|IOC|FOK|POST\nM id B|S qty IOC|FOK\n"
              "C id\nR id tick qty\nBOOK\nCHECK\nQUIT");
    char line[512];
    while(std::fgets(line,sizeof(line),stdin)) {
        if(!std::strchr(line,'\n') && !std::feof(stdin)) {
            int ch;while((ch=std::getchar())!='\n' && ch!=EOF){}std::puts("PARSE ERROR: line too long");continue;
        }
        Tokens t=tokenize(line);if(!t.n)continue;
        if(!t.overflow && t.n==1) {
            if(eq(t.v[0],"QUIT"))break;
            if(eq(t.v[0],"BOOK")){print_depth(book);continue;}
            if(eq(t.v[0],"CHECK")){std::puts(book.validate()?"OK":"INVALID");continue;}
        }
        Command c;if(!parse_command(t,c)){std::puts("PARSE ERROR");continue;}
        book.process(c,events);for(const auto& e:events)print_event(e);
    }
}

// A reproducible throughput microbenchmark, NOT a tail-latency measurement.
// Fixed add/add/match/cancel cycle, one price, low occupancy. Construction and
// warm-up are outside timing; event checksum keeps work observable. There is no
// I/O in the loop. To study p99/p99.9, add realistic replay, bursts, queueing,
// clock-overhead calibration, affinity and hardware-specific measurements.
void benchmark(std::uint64_t cycles) {
    static DemoBook book;static DemoBook::EventBatch out;
    std::uint64_t checksum=0;
    auto cycle=[&](std::uint64_t k) {
        Id base=k*3+1;
        Command commands[]={limit(base,Side::Buy,100,10),limit(base+1,Side::Buy,100,20),
                            market(base+2,Side::Sell,15),cancel(base+1)};
        for(const auto& c:commands) {
            book.process(c,out);
            for(const auto& e:out)checksum+=e.id+e.maker_id+e.qty+e.leaves+e.canceled;
        }
    };
    for(std::uint64_t i=0;i<10000;++i)cycle(i);
    const auto start=std::chrono::steady_clock::now();
    for(std::uint64_t i=0;i<cycles;++i)cycle(i);
    const auto end=std::chrono::steady_clock::now();
    const double ns=std::chrono::duration<double,std::nano>(end-start).count();
    CHECK(book.size()==0 && book.validate());
    std::printf("single-thread add/add/match/cancel; cycles=%llu commands=%llu\n"
                "elapsed=%.3f ms mean=%.2f ns/command throughput=%.2f Mcommands/s checksum=%llu\n"
                "book bytes=%zu batch bytes=%zu; includes event materialization, excludes queue/IO\n"
                "Synthetic low-occupancy throughput only; not p99 latency or a deployment guarantee.\n",
                (unsigned long long)cycles,(unsigned long long)(cycles*4),ns/1e6,ns/double(cycles*4),
                double(cycles*4)*1000.0/ns,(unsigned long long)checksum,sizeof(book),sizeof(out));
}
} // namespace app

int main(int argc,char** argv) {
    using namespace app;
    if(argc==1 || (argc==2 && eq(argv[1],"--demo"))){demo();return 0;}
    if(argc==2 && eq(argv[1],"--test")) {
        container_tests();std::puts("custom container tests passed");
        deterministic_tests();std::puts("matching edge cases passed");
        randomized_tests();std::puts("300000 reference comparisons passed");
        queue_tests();std::puts("SPSC million-message and backpressure tests passed");return 0;
    }
    if(argc==2 && eq(argv[1],"--queue-test")){queue_tests();std::puts("queue tests passed");return 0;}
    if(argc==2 && eq(argv[1],"--cli")){cli();return 0;}
    if((argc==2 || argc==3) && eq(argv[1],"--bench")) {
        std::uint64_t cycles=250000;
        if(argc==3 && (!unsigned_number(argv[2],1000000000ULL,cycles)||!cycles)) {
            std::fputs("benchmark cycles must be 1..1000000000\n",stderr);return 2;
        }
        benchmark(cycles);return 0;
    }
    std::fputs("Usage: engine [--demo | --test | --queue-test | --cli | --bench [cycles]]\n",stderr);
    return 2;
}
#endif // HFT_NO_MAIN
