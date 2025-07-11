//
// Created by Giorgio Vinciguerra on 26/01/25.
//

#pragma once

#include <immintrin.h>
#include <cstdint>
#include <cassert>

namespace lsf::bits {

const uint64_t lo_set[] = {
    0x0000000000000000ULL,
    0x0000000000000001ULL,
    0x0000000000000003ULL,
    0x0000000000000007ULL,
    0x000000000000000FULL,
    0x000000000000001FULL,
    0x000000000000003FULL,
    0x000000000000007FULL,
    0x00000000000000FFULL,
    0x00000000000001FFULL,
    0x00000000000003FFULL,
    0x00000000000007FFULL,
    0x0000000000000FFFULL,
    0x0000000000001FFFULL,
    0x0000000000003FFFULL,
    0x0000000000007FFFULL,
    0x000000000000FFFFULL,
    0x000000000001FFFFULL,
    0x000000000003FFFFULL,
    0x000000000007FFFFULL,
    0x00000000000FFFFFULL,
    0x00000000001FFFFFULL,
    0x00000000003FFFFFULL,
    0x00000000007FFFFFULL,
    0x0000000000FFFFFFULL,
    0x0000000001FFFFFFULL,
    0x0000000003FFFFFFULL,
    0x0000000007FFFFFFULL,
    0x000000000FFFFFFFULL,
    0x000000001FFFFFFFULL,
    0x000000003FFFFFFFULL,
    0x000000007FFFFFFFULL,
    0x00000000FFFFFFFFULL,
    0x00000001FFFFFFFFULL,
    0x00000003FFFFFFFFULL,
    0x00000007FFFFFFFFULL,
    0x0000000FFFFFFFFFULL,
    0x0000001FFFFFFFFFULL,
    0x0000003FFFFFFFFFULL,
    0x0000007FFFFFFFFFULL,
    0x000000FFFFFFFFFFULL,
    0x000001FFFFFFFFFFULL,
    0x000003FFFFFFFFFFULL,
    0x000007FFFFFFFFFFULL,
    0x00000FFFFFFFFFFFULL,
    0x00001FFFFFFFFFFFULL,
    0x00003FFFFFFFFFFFULL,
    0x00007FFFFFFFFFFFULL,
    0x0000FFFFFFFFFFFFULL,
    0x0001FFFFFFFFFFFFULL,
    0x0003FFFFFFFFFFFFULL,
    0x0007FFFFFFFFFFFFULL,
    0x000FFFFFFFFFFFFFULL,
    0x001FFFFFFFFFFFFFULL,
    0x003FFFFFFFFFFFFFULL,
    0x007FFFFFFFFFFFFFULL,
    0x00FFFFFFFFFFFFFFULL,
    0x01FFFFFFFFFFFFFFULL,
    0x03FFFFFFFFFFFFFFULL,
    0x07FFFFFFFFFFFFFFULL,
    0x0FFFFFFFFFFFFFFFULL,
    0x1FFFFFFFFFFFFFFFULL,
    0x3FFFFFFFFFFFFFFFULL,
    0x7FFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL
};


/** Extract contiguous bits from a 64-bit integer. */
inline uint64_t bextr(uint64_t word, unsigned int offset, unsigned int length) {
#ifdef __BMI__
    return _bextr_u64(word, offset, length);
#else
    return (word >> offset) & lo_set[length];
#endif
}

/** Reads the specified number of bits (must be < 58) from the given position. */
inline uint64_t read_int(const uint64_t *data, uint64_t bit_offset, uint8_t length) {
    // assert(length < 58);
    // auto ptr = reinterpret_cast<const char*>(data);
    // auto word = *(reinterpret_cast<const uint64_t *>(ptr + bit_offset / 8));
    // return bextr(word, bit_offset % 8, length);
    // from https://github.com/vigna/sux/
    auto start_word = bit_offset / 64;
    auto start_bit = bit_offset % 64;
    auto total_offset = start_bit + length;
    const uint64_t result = data[start_word] >> start_bit;
    return (total_offset <= 64 ? result : result | data[start_word + 1] << (64 - start_bit)) & ((1ULL << length) - 1);
}

/** Writes the specified value at the given position. */
inline void write_int(uint64_t *data, uint64_t bit_offset, int length, uint64_t value) {
    if (length >= 64)
        throw std::runtime_error("Code length exceeds 64 bits");

    // from https://github.com/vigna/sux/
    auto start_word = bit_offset / 64;
    auto end_word = (bit_offset + length - 1) / 64;
    auto start_bit = bit_offset % 64;

    if (start_word == end_word) {
        data[start_word] &= ~(((1ULL << length) - 1) << start_bit);
        data[start_word] |= value << start_bit;
    } else {
        // Here start_bit > 0.
        data[start_word] &= (1ULL << start_bit) - 1;
        data[start_word] |= value << start_bit;
        data[end_word] &= -(1ULL << (length - 64 + start_bit));
        data[end_word] |= value >> (64 - start_bit);
    }
}

// https://graphics.stanford.edu/~seander/bithacks.html#BitReverseTable
static const unsigned char BitReverseTable256[256] = {
#   define R2(n)     n,     n + 2*64,     n + 1*64,     n + 3*64
#   define R4(n) R2(n), R2(n + 2*16), R2(n + 1*16), R2(n + 3*16)
#   define R6(n) R4(n), R4(n + 2*4 ), R4(n + 1*4 ), R4(n + 3*4 )
    R6(0), R6(2), R6(1), R6(3)
#undef R2
#undef R4
#undef R6
};

/** Reverses the bits of the given code. */
static uint64_t bit_reverse(uint64_t code, int length) {
    if (length <= 8) {
        return BitReverseTable256[code] >> (8 - length);
    }
    if (length <= 16) {
        auto out = BitReverseTable256[code & 0xFF] << 8;
        out |= BitReverseTable256[(code >> 8) & 0xFF];
        return out >> (16 - length);
    }
    uint64_t result = 0;
    for (int i = 0; i < length; ++i) {
        result = (result << 1) | (code & 1);
        code >>= 1;
    }
    return result;
// #if __has_builtin(__builtin_bitreverse64)
//     return __builtin_bitreverse64(code) >> (64 - length);
// #else
//     constexpr auto s = sizeof(code);
//     unsigned char out[s];
//     unsigned char in[s];
//     std::memcpy(in, &code, s);
//     for (int i = 0; i < s; ++i) {
//         out[i] = BitReverseTable256[in[s - 1 - i]];
//     }
//     std::memcpy(&code, out, s);
//     return code >> (64 - length);
// #endif
}

}