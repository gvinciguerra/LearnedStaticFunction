//
// Created by Giorgio Vinciguerra on 26/01/25.
//

#pragma once

#include <immintrin.h>
#include <cstdint>
#include <cassert>

namespace learnedretrieval::bits {

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
    assert(length < 58);
    auto ptr = reinterpret_cast<const char*>(data);
    auto word = *(reinterpret_cast<const uint64_t *>(ptr + bit_offset / 8));
    return bextr(word, bit_offset % 8, length);
}

/** Writes the specified value at the given position. */
inline void write_int(uint64_t *data, uint64_t bit_offset, int length, uint64_t value) {
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

}