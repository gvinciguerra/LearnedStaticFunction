#pragma once

#pragma once

#include <algorithm>
#include <cassert>
#include <vector>
#include <ranges>
#include <numeric>
#include "bits.hpp"

/*
 * Variable Length Retrieval with filter uses a filter at each node of the prefix free code
 * Note that the length of the filter bits might be zero meaning that no filter is used
 */


namespace learnedretrieval {
    struct FilterCode {
        using LengthType = uint64_t;
        uint64_t correctedCode;
        uint64_t filterCode;
        LengthType codeLength;
        LengthType correctionLength;
        LengthType filterLength;
    };

    class FilterLengthStrategy1 {
        static uint64_t getFilterBits(float probability) {
            return 1;
        }
    };

    template<typename Coder, typename FilterLengthStrategy = FilterLengthStrategy1, typename Symbol = uint32_t, typename Frequency = float>
    class FilterCoding {
        FilterCode encode_once(const std::span<Frequency> &f, Symbol symbol) {

        }

        Symbol decode_once(const std::span<Frequency> &f, const uint64_t *corrected_code_data, size_t &corrected_code_bit_offset, const uint64_t *filter_code_data, size_t &filter_code_bit_offset) {
            Coder coder(f);
            while (!coder.hasFinished()) {
                float probability = coder.getNextProbability();
                uint64_t filterBitLength = FilterLengthStrategy::getFilterBits(probability);
                uint64_t filterBits = 0; //ToDo read next filter bits
                if(filterBits) {
                    coder.nextBit(true);
                } else {
                    bool nextBit = 0; // ToDo get next code bit
                    coder.nextBit(nextBit);
                }
            }
            return coder.getResult();
        }
    };
}