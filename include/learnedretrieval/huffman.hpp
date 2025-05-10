#pragma once

#include <algorithm>
#include <cassert>
#include <vector>
#include <ranges>
#include <numeric>
#include <boost/sort/sort.hpp>
#include "bits.hpp"


namespace learnedretrieval {
template<typename Symbol = uint32_t, typename Frequency = float>
class Huffman {
    std::vector<Symbol> symbols;
    std::span<Frequency> frequencies;
    std::vector<int> lengths;

public:
    struct Code {
        uint64_t code; ///< The value of the code.
        int length; ///< The length of the code in bits.
    };

    Huffman() = default;

    template<typename Frequencies>
    explicit Huffman(Frequencies &f) : symbols(f.size()), frequencies(f), lengths(f.size()) {
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

    void compute_code_lengths() {
        std::iota(symbols.begin(), symbols.end(), 0);

        for (auto i = 0; i < frequencies.size(); ++i) {
            if constexpr (std::is_floating_point_v<Frequency>) {
                using length_type = typename decltype(lengths)::value_type;
                lengths[i] = static_cast<length_type>((long double)(frequencies[i]) * std::numeric_limits<length_type>::max());
            } else {
                lengths[i] = frequencies[i];
            }
        }

        // Algorithm 2 from https://dl.acm.org/doi/pdf/10.1145/3342555 to compute code lengths
        auto zipv = std::ranges::views::zip(symbols, lengths);
        auto comp = [](const auto &x, const auto &y) { return std::get<1>(x) > std::get<1>(y); };
        boost::sort::pdqsort_branchless(zipv.begin(), zipv.end(), comp);

        // Phase 1
        auto n = (int) symbols.size();
        auto leaf = n - 1;
        auto root = n - 1;
        for (auto next = n - 1; next > 0; --next) {
            if (leaf < 0 || (root > next && lengths[root] < lengths[leaf])) {
                lengths[next] = lengths[root];
                lengths[root] = next;
                --root;
            } else {
                lengths[next] = lengths[leaf];
                --leaf;
            }
            if (leaf < 0 || (root > next && lengths[root] < lengths[leaf])) {
                lengths[next] += lengths[root];
                lengths[root] = next;
                --root;
            } else {
                lengths[next] += lengths[leaf];
                --leaf;
            }
        }

        // Phase 2
        lengths[1] = 0;
        for (auto next = 2; next < n; ++next)
            lengths[next] = lengths[lengths[next]] + 1;

        // Phase 3
        auto avail = 1;
        auto used = 0;
        auto depth = 0;
        auto next = 0;
        root = 1;
        while (avail > 0) {
            while (root < n && lengths[root] == depth) {
                ++used;
                ++root;
            }
            while (avail > used) {
                lengths[next] = std::min(depth, 63);
                ++next;
                --avail;
            }
            avail = 2 * used;
            ++depth;
            used = 0;
        }
    }

    Code encode_internal(Symbol symbol) {
        uint64_t code = 0;
        size_t i;
        for (i = 1; i < symbols.size() && symbol != symbols[i - 1]; ++i) {
            code = (code + 1) << (lengths[i] - lengths[i - 1]);
        }
        auto length = lengths[i - 1];
        return {bits::bit_reverse(code, length), length};
    }

    Symbol decode_internal(const uint64_t *data, size_t &bit_offset) {
        auto buffer_length = lengths[0];
        uint64_t code = 0;
        uint64_t buffer = bits::read_int(data, bit_offset, buffer_length);
        buffer = bits::bit_reverse(buffer, buffer_length);
        bit_offset += buffer_length;

        if (code == buffer)
            return symbols[0];

        for (size_t i = 1; i < symbols.size(); ++i) {
            while (buffer_length < lengths[i]) {
                buffer = (buffer << 1) | ((data[bit_offset / 64] >> (bit_offset % 64)) & 1);
                ++bit_offset;
                ++buffer_length;
            }
            code = (code + 1) << (lengths[i] - lengths[i - 1]);
            if (code == buffer)
                return symbols[i];
        }

        throw std::runtime_error("Invalid code");
        // throw std::runtime_error("Invalid code");
        // auto max_length = lengths.back();
        // uint64_t buffer = bits::bit_reverse(bits::read_int(data, bit_offset, max_length), max_length);
        // uint64_t code = 0;
        // int length = 0;
        // int cumulative_count = 0;
        // int i = 0;
        // while (buffer >= code) {
        //     auto j = i;
        //     while (lengths[i] == length)
        //         ++i;
        //     auto length_counts = i - j;
        //     auto next_code = code + length_counts * (uint64_t(1) << (max_length - length));
        //     if (buffer < next_code)
        //         break;
        //     code = next_code;
        //     cumulative_count += length_counts;
        //     ++length;
        // }
        //
        // auto symbol_id = (int) ((buffer - code) >> (max_length - length)); // among those with the same length
        // bit_offset += length;
        // return symbols[cumulative_count + symbol_id];
    }
};
}
