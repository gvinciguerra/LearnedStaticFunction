#pragma once

#include <algorithm>
#include <cassert>
#include <vector>
#include <ranges>
#include <numeric>
#include "bits.hpp"

namespace learnedretrieval {
template<typename Symbol = uint32_t, typename Frequency = float>
class Huffman {
    std::vector<Symbol> symbols;
    std::vector<Frequency> frequencies;
    struct Code {
        uint64_t code;  ///< The value of the code.
        int length; ///< The length of the code in bits.
    };

public:

    Huffman() = default;

    template <typename Frequencies>
    explicit Huffman(Frequencies &freqs) : symbols(freqs.size()), frequencies(freqs) {
        std::iota(symbols.begin(), symbols.end(), 0);
        compute_code_lengths(symbols, frequencies);
    }

    template<typename Symbols, typename Frequencies>
    static Code encode_once(Symbols &tmp_symbols, Frequencies &tmp_frequencies, Symbol symbol) {
        assert(symbol < tmp_frequencies.size() && tmp_frequencies.size() == tmp_symbols.size());
        std::iota(tmp_symbols.begin(), tmp_symbols.end(), 0);
        compute_code_lengths(tmp_symbols, tmp_frequencies);
        return encode_internal(tmp_symbols, tmp_frequencies, symbol);
    }

    template<typename Symbols, typename Frequencies>
    static Symbol decode_once(Symbols &tmp_symbols, Frequencies &tmp_frequencies, const uint64_t *data, size_t &bit_offset) {
        assert(tmp_frequencies.size() == tmp_symbols.size());
        std::iota(tmp_symbols.begin(), tmp_symbols.end(), 0);
        compute_code_lengths(tmp_symbols, tmp_frequencies);
        return decode_internal(tmp_frequencies, data, bit_offset);
    }

    Code encode(Symbol symbol) {
        return encode_internal(symbols, frequencies, symbol);
    }

    void encode_write(Symbol symbol, uint64_t *data, size_t &bit_offset) {
        auto code = encode_internal(symbols, frequencies, symbol);
        bits::write_int(data, bit_offset, code.length, code.code);
        bit_offset += code.length;
    }

    Symbol decode(const uint64_t *data, size_t &bit_offset) {
        return decode_internal(symbols, frequencies, data, bit_offset);
    }

private:

    template<typename Symbols, typename Frequencies>
    static Code encode_internal(const Symbols &symbols, const Frequencies &frequencies, Symbol symbol) {
        uint64_t code = 0;
        size_t i;
        for (i = 1; i < symbols.size() && symbol != symbols[i - 1]; ++i) {
            code = (code + 1) << int(frequencies[i] - frequencies[i - 1]);
            // std::cout << i << " -> " << bit_reverse(code, frequencies[i]) << " " << frequencies[i] << std::endl;
        }

        auto length = (int) frequencies[i - 1];
        return {bit_reverse(code, length), length};
    }

    template<typename Symbols, typename Frequencies>
    static Symbol decode_internal(const Symbols &symbols, const Frequencies &frequencies, const uint64_t *data, size_t &bit_offset) {
        auto buffer_length = frequencies[0];
        uint64_t code = 0;
        uint64_t buffer = bits::read_int(data, bit_offset, buffer_length);
        buffer = bit_reverse(buffer, buffer_length);
        bit_offset += buffer_length;

        if (code == buffer)
            return symbols[0];

        for (size_t i = 1; i < symbols.size(); ++i) {
            while (buffer_length < frequencies[i]) {
                buffer = (buffer << 1) | ((data[bit_offset / 64] >> (bit_offset % 64)) & 1);
                ++bit_offset;
                ++buffer_length;
            }
            code = (code + 1) << int(frequencies[i] - frequencies[i - 1]);
            if (code == buffer)
                return symbols[i];
        }

        throw std::runtime_error("Invalid code");
    }

    /** Reverses the bits of the given code. */
    static uint64_t bit_reverse(uint64_t code, int length) {
        uint64_t result = 0;
        for (int i = 0; i < length; ++i) {
            result = (result << 1) | (code & 1);
            code >>= 1;
        }
        return result;
    }

    /** Computes the code lengths for the given frequencies. Upon return, the frequencies vector will contain the code
     *  lengths in increasing order, and symbols are rearranged accordingly. */
    template<typename Symbols, typename Frequencies>
    static void compute_code_lengths(Symbols &symbols, Frequencies &freqs) {
        assert(freqs.size() == symbols.size());

        // Algorithm 2 from https://dl.acm.org/doi/pdf/10.1145/3342555 to compute code lengths
        std::ranges::sort(std::ranges::views::zip(symbols, freqs), [](const auto &a, const auto &b) {
            return std::get<1>(a) > std::get<1>(b);
        });

        // Phase 1
        auto n = (int) symbols.size();
        auto leaf = n - 1;
        auto root = n - 1;
        for (auto next = n - 1; next > 0; --next) {
            if (leaf < 0 || (root > next && freqs[root] < freqs[leaf])) {
                freqs[next] = freqs[root];
                freqs[root] = next;
                --root;
            } else {
                freqs[next] = freqs[leaf];
                --leaf;
            }
            if (leaf < 0 || (root > next && freqs[root] < freqs[leaf])) {
                freqs[next] += freqs[root];
                freqs[root] = next;
                --root;
            } else {
                freqs[next] += freqs[leaf];
                --leaf;
            }
        }

        // Phase 2
        freqs[1] = 0;
        for (auto next = 2; next < n; ++next)
            freqs[next] = freqs[freqs[next]] + 1;

        // Phase 3
        auto avail = 1;
        auto used = 0;
        auto depth = 0;
        auto next = 0;
        root = 1;
        while (avail > 0) {
            while (root < n && freqs[root] == depth) {
                ++used;
                ++root;
            }
            while (avail > used) {
                freqs[next] = depth;
                ++next;
                --avail;
            }
            avail = 2 * used;
            ++depth;
            used = 0;
        }
    }
};

}