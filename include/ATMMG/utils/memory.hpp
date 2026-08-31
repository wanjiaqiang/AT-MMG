#pragma once

#include <immintrin.h>
#if !defined(_WIN32)
#include <sys/mman.h>
#endif

#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <type_traits>
#if defined(_MSC_VER)
#include <malloc.h>
#endif

#include "ATMMG/utils/tools.hpp"

namespace ATMMG::memory {
#if defined(_MSC_VER)
#define PORTABLE_ALIGN32 __declspec(align(32))
#define PORTABLE_ALIGN64 __declspec(align(64))
#else
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))
#endif

inline void* aligned_malloc_portable(size_t alignment, size_t nbytes) {
#if defined(_MSC_VER)
    return _aligned_malloc(nbytes, alignment);
#else
    return std::aligned_alloc(alignment, nbytes);
#endif
}

inline void aligned_free_portable(void* ptr) {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

inline void advise_hugepage([[maybe_unused]] void* ptr, [[maybe_unused]] size_t nbytes) {
#if !defined(_WIN32)
    madvise(ptr, nbytes, MADV_HUGEPAGE);
#endif
}

template <typename T, size_t Alignment = 64, bool HugePage = false>
class AlignedAllocator {
   private:
    static_assert(Alignment >= alignof(T));

   public:
    using value_type = T;

    template <class U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    constexpr AlignedAllocator() noexcept = default;

    constexpr AlignedAllocator(const AlignedAllocator&) noexcept = default;

    template <typename U>
    constexpr explicit AlignedAllocator(AlignedAllocator<U, Alignment> const&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t n) {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }

        auto nbytes = round_up_to_multiple_of<size_t>(n * sizeof(T), Alignment);
        auto* ptr = aligned_malloc_portable(Alignment, nbytes);
        if (HugePage) {
            advise_hugepage(ptr, nbytes);
        }
        return reinterpret_cast<T*>(ptr);
    }

    void deallocate(T* ptr, [[maybe_unused]] std::size_t n) {
        aligned_free_portable(ptr);
    }
};

template <typename T>
struct Allocator {
   public:
    using value_type = T;

    constexpr Allocator() noexcept = default;

    template <typename U>
    explicit constexpr Allocator(const Allocator<U>&) noexcept {}

    [[nodiscard]] constexpr T* allocate(std::size_t n) { return ::new T[n]; }

    constexpr void deallocate(T* ptr, [[maybe_unused]] size_t n) noexcept {
        ::delete[] ptr;
    }

    // Intercept zero-argument construction to do default initialization.
    template <typename U>
    void construct(U* ptr) noexcept(std::is_nothrow_default_constructible_v<U>) {
        ::new (static_cast<void*>(ptr)) U;
    }
};

template <size_t Alignment, typename T, bool HugePage = false>
inline T* align_allocate(size_t nbytes) {
    auto size = round_up_to_multiple_of<size_t>(nbytes, Alignment);
    void* ptr = aligned_malloc_portable(Alignment, size);
    if (HugePage) {
        advise_hugepage(ptr, size);
    }
    return static_cast<T*>(ptr);
}

static inline void prefetch_l1(const void* addr) {
#if defined(__SSE2__) || defined(_MSC_VER)
    _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0);
#else
    __builtin_prefetch(addr, 0, 3);
#endif
}

static inline void prefetch_l2(const void* addr) {
#if defined(__SSE2__) || defined(_MSC_VER)
    _mm_prefetch((const char*)addr, _MM_HINT_T1);
#else
    __builtin_prefetch(addr, 0, 2);
#endif
}

inline void mem_prefetch_l1(const char* ptr, size_t num_lines) {
    switch (num_lines) {
        default:
            [[fallthrough]];
        case 20:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 19:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 18:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 17:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 16:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 15:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 14:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 13:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 12:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 11:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 10:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 9:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 8:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 7:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 6:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 5:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 4:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 3:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 2:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 1:
            prefetch_l1(ptr);
            ptr += 64;
            [[fallthrough]];
        case 0:
            break;
    }
}

inline void mem_prefetch_l2(const char* ptr, size_t num_lines) {
    switch (num_lines) {
        default:
            [[fallthrough]];
        case 20:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 19:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 18:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 17:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 16:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 15:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 14:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 13:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 12:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 11:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 10:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 9:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 8:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 7:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 6:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 5:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 4:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 3:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 2:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 1:
            prefetch_l2(ptr);
            ptr += 64;
            [[fallthrough]];
        case 0:
            break;
    }
}
}  // namespace ATMMG::memory
