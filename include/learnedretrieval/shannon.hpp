#pragma once

#include <algorithm>
#include <cassert>
#include <vector>
#include <limits>
#include <boost/sort/sort.hpp>
#include "bits.hpp"


namespace learnedretrieval {
template<typename Symbol = uint32_t, typename Frequency = float>
class Shannon {
    static_assert(std::is_floating_point_v<Frequency>, "Frequency must be an floating type");
    std::span<Frequency> frequencies;
    std::vector<int> lengths;
    std::vector<int> length_counts;
    int max_length;

public:
    struct Code {
        uint64_t code; ///< The value of the code.
        int length; ///< The length of the code in bits.
    };

    Shannon() = default;

    template<typename Frequencies>
    explicit Shannon(Frequencies &f) : frequencies(f), lengths(f.size()), length_counts(65) {
        compute_code_lengths();
    }

    void reset(const std::span<Frequency> &f) {
        assert(f.size() == frequencies.size());
        frequencies = f;
        compute_code_lengths();
    }

    void encode_write(Symbol symbol, uint64_t *data, size_t &bit_offset) {
        auto code = encode_internal(symbol);
        bits::write_int(data, bit_offset, code.length, code.code);
        bit_offset += code.length;
    }

    Code encode(Symbol symbol) {
        return encode_internal(symbol);
    }

    Symbol decode(const uint64_t *data, size_t &bit_offset) {
        return decode_internal(data, bit_offset);
    }

private:
    Code encode_internal(Symbol symbol) {
        auto length = lengths[symbol];
        uint16_t id = 0; // symbol code id among those with the same length
        for (size_t i = 0; i < symbol; ++i) {
            if (lengths[i] == lengths[symbol])
                id++;
        }

        uint64_t code = 0;
        for (auto l = 1; l <= max_length; ++l) {
            code = (code + length_counts[l - 1]) << 1;
            if (l == length) {
                code += id;
                return {bits::bit_reverse(code, length), length};
            }
        }

        throw std::runtime_error("Invalid symbol");
    }

    Symbol decode_internal(const uint64_t *data, size_t &bit_offset) {
        uint64_t buffer = bits::bit_reverse(bits::read_int(data, bit_offset, max_length), max_length);
        uint64_t code = 0;
        int length = 0;
        while (buffer >= code) {
            auto next_code = code + length_counts[length] * (uint64_t(1) << (max_length - length));
            if (buffer < next_code)
                break;
            code = next_code;
            ++length;
        }

        auto symbol_id = (int) ((buffer - code) >> (max_length - length)); // among those with the same length
        size_t i = 0;
        while (symbol_id >= 0) {
            symbol_id -= lengths[i] == length;
            ++i;
        }

        bit_offset += length;
        return i - 1;
    }

    void compute_code_lengths() {
        max_length = 0;
        for (size_t i = 0; i < frequencies.size(); ++i) {
            int res;
            std::frexp(frequencies[i], &res);
            res = frequencies[i] == 0 ? 63 : std::clamp<int>(1 + -res, 1, 63);
            lengths[i] = res;
            // lengths[i] = frequencies[i] ? (int) std::ceil(-std::log2(frequencies[i])) : 63;
            // lengths[i] = std::clamp<int>(lengths[i], 1, 63);
            max_length = std::max<int>(max_length, lengths[i]);
        }
        std::fill(length_counts.begin(), length_counts.begin() + max_length + 1, 0);
        for (size_t i = 0; i < lengths.size(); ++i)
            ++length_counts[lengths[i]];
    }
};
}
