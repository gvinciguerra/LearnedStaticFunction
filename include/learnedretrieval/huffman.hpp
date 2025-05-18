#pragma once

#include <algorithm>
#include <cassert>
#include <vector>
#include <ranges>
#include <numeric>
#include <boost/sort/sort.hpp>
#include <ips2ra.hpp>
#include "bits.hpp"


namespace learnedretrieval {
template<typename Symbol = uint32_t, typename Frequency = float>
class Huffman {
    using LengthType = uint8_t;
    struct SymbolLength {
        Symbol symbol;
        LengthType length;
    };
    std::vector<SymbolLength> table;
    std::vector<SymbolLength> temp_table;
    std::array<uint32_t, 256> sort_counts;
    std::span<Frequency> frequencies;

public:
    struct Code {
        uint64_t code; ///< The value of the code.
        LengthType length; ///< The length of the code in bits.
    };

    Huffman() = default;

    template<typename Frequencies>
    explicit Huffman(Frequencies &f) : table(f.size()), temp_table(f.size()), frequencies(f) {
        compute_code_lengths();
    }

    explicit Huffman(size_t size) : table(size), temp_table(size) { }

    void reset(const std::span<Frequency> &f) {
        assert(f.size() == table.size());
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

    Code encode_once(const std::span<Frequency> &f, Symbol symbol) {
        if constexpr (std::is_floating_point_v<Frequency>) {
            if (f[symbol] > 0.5)
                return {0, 1};
        }
        reset(f);
        return encode_internal(symbol);
    }

    Symbol decode_once(const std::span<Frequency> &f, const uint64_t *data, size_t &bit_offset) {
        if constexpr (std::is_floating_point_v<Frequency>) {
            auto max_frequency = std::max_element(f.begin(), f.end());
            if (*max_frequency > 0.5 && ((data[bit_offset / 64] >> (bit_offset % 64)) & 1) == 0) {
                ++bit_offset;
                return max_frequency - f.begin();
            }
        }
        reset(f);
        return decode_internal(data, bit_offset);
    }

private:

    void compute_code_lengths() {
        for (Symbol i = 0; i < frequencies.size(); ++i) {
            if constexpr (std::is_floating_point_v<Frequency>) {
                constexpr double scaling_factor = std::numeric_limits<LengthType>::max();
                table[i] = {i, static_cast<LengthType>(frequencies[i] * scaling_factor)};
            } else {
                table[i] = {i, frequencies[i]};
            }
        }

        // Algorithm 2 from https://dl.acm.org/doi/pdf/10.1145/3342555 to compute code lengths
        auto comp = [](const auto &x, const auto &y) { return x.length > y.length; };
        if constexpr (std::is_same_v<LengthType, uint8_t>) {
            if (table.size() < 32) {
                boost::sort::pdqsort_branchless(table.begin(), table.end(), comp);
            } else {
                std::fill(sort_counts.begin(), sort_counts.end(), 0);
                for (auto it = table.begin(); it != table.end(); ++it)
                    ++sort_counts[it->length];
                std::partial_sum(sort_counts.begin(), sort_counts.end(), sort_counts.begin());
                for (auto it = table.end() - 1; it >= table.begin(); --it)
                    temp_table[temp_table.size() - 1 - --sort_counts[it->length]] = *it;
                table.swap(temp_table);
            }
        } else {
            boost::sort::pdqsort_branchless(table.begin(), table.end(), comp);
            // std::sort(table.begin(), table.end(), comp);
            // ips2ra::sort(table.rbegin(), table.rend(), [&] (const auto &x) { return x.length; });
        }

        // Phase 1
        auto n = (int) table.size();
        auto leaf = n - 1;
        auto root = n - 1;
        for (auto next = n - 1; next > 0; --next) {
            if (leaf < 0 || (root > next && table[root].length < table[leaf].length)) {
                table[next].length = table[root].length;
                table[root].length = next;
                --root;
            } else {
                table[next].length = table[leaf].length;
                --leaf;
            }
            if (leaf < 0 || (root > next && table[root].length < table[leaf].length)) {
                table[next].length += table[root].length;
                table[root].length = next;
                --root;
            } else {
                table[next].length += table[leaf].length;
                --leaf;
            }
        }

        // Phase 2
        table[1].length = 0;
        for (auto next = 2; next < n; ++next)
            table[next].length = table[table[next].length].length + 1;

        // Phase 3
        auto avail = 1;
        auto used = 0;
        auto depth = 0;
        auto next = 0;
        root = 1;
        while (avail > 0) {
            while (root < n && table[root].length == depth) {
                ++used;
                ++root;
            }
            while (avail > used) {
                table[next].length = std::min(depth, 63);
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
        for (i = 1; i < table.size() && symbol != table[i - 1].symbol; ++i) {
            code = (code + 1) << (table[i].length - table[i - 1].length);
        }
        auto length = table[i - 1].length;
        return {bits::bit_reverse(code, length), length};
    }

    Symbol decode_internal(const uint64_t *data, size_t &bit_offset) {
        auto buffer_length = table[0].length;
        uint64_t code = 0;
        uint64_t buffer = bits::read_int(data, bit_offset, buffer_length);
        buffer = bits::bit_reverse(buffer, buffer_length);
        bit_offset += buffer_length;

        if (code == buffer)
            return table[0].symbol;

        for (size_t i = 1; i < table.size(); ++i) {
            while (buffer_length < table[i].length) {
                buffer = (buffer << 1) | ((data[bit_offset / 64] >> (bit_offset % 64)) & 1);
                ++bit_offset;
                ++buffer_length;
            }
            code = (code + 1) << (table[i].length - table[i - 1].length);
            if (code == buffer)
                return table[i].symbol;
        }

        throw std::runtime_error("Invalid code");
        // auto max_length = table.back().length;
        // uint64_t buffer = bits::bit_reverse(bits::read_int(data, bit_offset, max_length), max_length);
        // uint64_t code = 0;
        // int length = 0;
        // int cumulative_count = 0;
        // int i = 0;
        // while (buffer >= code) {
        //     auto j = i;
        //     while (table[i].length == length)
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
        // return table[cumulative_count + symbol_id].symbol;
    }
};
}
