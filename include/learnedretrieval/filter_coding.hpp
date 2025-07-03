#pragma once

#include <algorithm>
#include <cassert>
#include <vector>
#include <ranges>
#include <numeric>
#include <queue>
#include <functional>
#include <iostream>
#include "bits.hpp"

/*
 * Variable Length Retrieval with filter uses a filter at each node of the prefix free code
 * Note that the length of the filter bits might be zero meaning that no filter is used
 *
 * encoding is a two step procedure
 * 1. construct the filter
 * 2. create the code which consists of bits that correct the filter decisions
 */


namespace learnedretrieval {
    constexpr float eps=0.0000001f;


    struct FilterCode {
        using LengthType = uint64_t;
        uint64_t code;
        LengthType length;
    };

    template<typename OtherFilter>
    class FilterLengthOnlyRootWrapper {
    public:
        static uint64_t getFilterBits(float probability, size_t level) {
            if (level > 0)
                return 0;
            return OtherFilter::getFilterBits(probability, level);
        }
    };

    class FilterLengthStrategyNoFilter {
    public:
        static uint64_t getFilterBits(float probability, size_t level) {
            return 0;
        }
    };

    int maxFilterCnt = 0;

    class FilterLengthStrategyOpt {
        static constexpr int MAX_FILTER_BITS = 20;
        static constexpr double PROBABILITY_THRESHOLDS[MAX_FILTER_BITS] = {0.333333, 0.2, 0.111111, 0.0588235, 0.030303,
                                                                           0.0153846, 0.00775194, 0.00389105,
                                                                           0.00194932, 0.00097561, 0.000488043,
                                                                           0.000244081, 0.000122055, 6.10314e-05,
                                                                           3.05166e-05, 1.52586e-05, 7.62934e-06, 3.81468e-06, 1.90734e-06, 9.53673e-07, /*4.76837e-07, 2.38419e-07, 1.19209e-07, 5.96046e-08, 2.98023e-08, 1.49012e-08, 7.45058e-09, 3.72529e-09, 1.86265e-09, 9.31323e-10, */};


    public:
        static uint64_t getFilterBits(float probability, size_t level) {
            //if(level>0)return 0;
            size_t bits = 0;
            while (bits < MAX_FILTER_BITS && PROBABILITY_THRESHOLDS[bits] > probability)
                bits++;
            if (bits == MAX_FILTER_BITS) {
                maxFilterCnt++;
            }
            return bits;
        }
    };

    template<typename Symbol = uint32_t, typename Frequency = float>
    class FilterFanoCoder {
        struct Elem {
            Frequency f;
            Symbol s;
            uint64_t code;

        public:
            bool getCodeBit(size_t pos) const {
                return (code >> pos) & 1;
            }
        };

        size_t n;
        std::vector<Elem> sorted;

        bool encodeBit;
        bool flipNext;
        size_t leftBound;
        size_t rightBound;
        float absoluteFreq;
        size_t center;
        Frequency lastCumFreq;
        size_t currentBitPos;

        static constexpr int BUCKETS = 10;
        static constexpr float MIN_P = 1.0f / float(1u << BUCKETS);
        std::array<size_t, BUCKETS> bucketCnt;

        Elem target;

        int getBucket(Frequency f) {
            int exp;
            frexp(f, &exp);
            if(f==0) {
                return BUCKETS-1;
            }
            if(f==1) {
                return 0;
            }
            return std::min(BUCKETS-1, -exp);
        }

    public:

        FilterFanoCoder(size_t n) : n(n) {
            sorted.resize(n);
        }

        template<bool encode = false>
        void init(const std::span<Frequency> &f, Symbol encodeSymbol = -1) {
            bucketCnt = {};

            flipNext = false;
            currentBitPos = BUCKETS;
            absoluteFreq = 0;
            lastCumFreq = 1.0;
            leftBound = 0;
            rightBound = f.size() - 1;


            for (Symbol i = 0; i < f.size(); ++i) {
                int bucket = getBucket(f[i]);
                bucketCnt[bucket]++;
            }
            size_t sum = 0;
            for (size_t i = 0; i < BUCKETS; ++i) {
                size_t offset = sum;
                sum += bucketCnt[i];
                bucketCnt[i] = offset;
            }
            for (Symbol i = 0; i < f.size(); ++i) {
                int b = getBucket(f[i]);
                size_t pos = bucketCnt[b]++;
                sorted[pos].s = i;
                sorted[pos].f = f[i];
            }

            uint64_t code = 0;
            for (size_t i = 0; i < f.size(); ++i) {
                sorted[i].code = std::min((uint64_t(2) << BUCKETS) - f.size() + i, code);
                if constexpr (encode) {
                    if (sorted[i].s == encodeSymbol) {
                        target = sorted[i];
                    }
                }
                code += uint64_t(1) << (BUCKETS - getBucket(sorted[i].f));
            }
            assert(sorted[n - 1].code < (uint64_t(2) << BUCKETS));
        }

        float getRelProbabilityAndAdvance() {
            absoluteFreq = 0;
            int index = leftBound;
            while (true) {
                const Elem e = sorted[index];
                if (e.getCodeBit(currentBitPos)) [[unlikely]] {
                    break;
                }
                absoluteFreq += e.f;
                index++;
                if (index == rightBound + 1) [[unlikely]] { // all 0 skip
                    absoluteFreq = 0;
                    index = leftBound;
                    currentBitPos--;
                }
            }
            center = index;
            float currentRelFeq = absoluteFreq / lastCumFreq;
            currentRelFeq= std::max(std::min(currentRelFeq, 1.0f-eps), eps);
            flipNext = currentRelFeq > 0.5f;
            if (flipNext) {
                currentRelFeq = 1.0f - currentRelFeq;
            }
            return currentRelFeq;
        }

        bool hasFinished() {
            return leftBound == rightBound;
        }

        void nextEncodeBit() {
            encodeBit = target.getCodeBit(currentBitPos) ^ flipNext;
            nextBit(encodeBit);
        }

        void nextBit(bool bit) {
            if (bit ^ flipNext) {
                leftBound = center;
                lastCumFreq = lastCumFreq - absoluteFreq;
            } else {
                rightBound = center - 1;
                lastCumFreq = absoluteFreq;
            }

            currentBitPos--;
        }

        bool getBit() {
            return encodeBit;
        }

        Symbol getResult() {
            return sorted[leftBound].s;
        }
    };


    template<typename Symbol = uint32_t, typename Frequency = float>
    class FilterHuffmanCoder {
    public:
        struct Node {
            Symbol s;
            size_t n1;
            size_t n2;
            size_t parent;
            bool bitRelParent;
            Frequency p;
            float relP;
            size_t index;
            bool leaf;
        };
        std::vector<Node> tree;
        Node root;
        Node currentDecodingNode;

        uint64_t encodeCode;
        bool lastEncBit;

        class Compare {
        public:
            bool operator()(Node a, Node b) {
                //return a.p > b.p;
                return a.index > b.index;
            }
        };

        FilterHuffmanCoder(size_t) {}


        template<bool encode = false>
        void init(const std::span<Frequency> &f, Symbol s = -1) {
            std::priority_queue<Node, std::vector<Node>, Compare> nodes;
            for (Symbol i = 0; i < f.size(); ++i) {
                Node n{i, 0, 0, 0, 0, f[i], 0, i, true};
                nodes.push(n);
                tree.push_back(n);
            }
            while (nodes.size() > 1) {
                Node a = nodes.top();
                nodes.pop();
                Node b = nodes.top();
                nodes.pop();

                if (a.p > b.p) {
                    Node c = a;
                    a = b;
                    b = c;
                }

                a.bitRelParent = 0;
                a.parent = tree.size();
                b.bitRelParent = 1;
                b.parent = tree.size();

                tree[a.index] = a;
                tree[b.index] = b;

                float relp;
                if(a.p+b.p==0){
                    relp=0.5f;
                } else {
                    relp=a.p / (a.p + b.p);
                }
                relp = std::max(std::min(relp, 1.0f-eps), eps);
                Node parent{0, a.index, b.index, 0, 0, a.p + b.p, relp, tree.size(), false};
                tree.push_back(parent);
                nodes.push(parent);
            }
            root = nodes.top();
            currentDecodingNode = root;

            if constexpr (encode) {
                Node current = tree[s];
                while (current.index != root.index) {
                    bool b = current.bitRelParent;
                    encodeCode <<= 1;
                    encodeCode |= b;
                    current = tree[current.parent];
                }
            }
        }

        float getRelProbabilityAndAdvance() {
            return currentDecodingNode.relP;
        }

        bool hasFinished() {
            return currentDecodingNode.leaf;
        }

        void nextEncodeBit() {
            lastEncBit = encodeCode & 1;
            nextBit(lastEncBit);
            encodeCode >>= 1;
        }

        void nextBit(bool bit) {
            currentDecodingNode = tree[bit ? currentDecodingNode.n2 : currentDecodingNode.n1];
        }

        bool getBit() {
            return lastEncBit;
        }

        Symbol getResult() {
            return currentDecodingNode.s;
        }
    };

    template<template<typename S, typename F> typename Coder, typename Symbol, typename Frequency>
    class Filter50PercentWrapper {
        Coder<Symbol, Frequency> coder;
        bool armed;
        bool exploded;
        Symbol armedSymbol;
        std::span<Frequency> fs;
    public:
        Filter50PercentWrapper(size_t n) : coder(n) {
        }

        template<bool encode = false>
        void init(const std::span<Frequency> &f, Symbol encodeSymbol = -1) {
            fs = f;
            exploded = false;
            armed = false;
            if constexpr (encode) {
                if (f[encodeSymbol] > 0.5f) {
                    armed = true;
                    armedSymbol = encodeSymbol;
                    return;
                }
            } else {
                for (Symbol i = 0; i < f.size(); i++) {
                    if (f[i] > 0.5f) {
                        armed = true;
                        armedSymbol = i;
                        return;
                    }
                }
            }
            coder.template init<encode>(f, encodeSymbol);
        }

        float getRelProbabilityAndAdvance() {
            if (armed) {
                return std::max(std::min(1.0f-fs[armedSymbol], 1.0f-eps), eps);
            }
            return coder.getRelProbabilityAndAdvance();
        }

        bool hasFinished() {
            return exploded || (!armed && coder.hasFinished());
        }

        void nextEncodeBit() {
            if(armed) {
                exploded = true;
            } else {
                coder.nextEncodeBit();
            }
        }

        void nextBit(bool bit) {
            if (armed) {
                if (bit) {
                    exploded = true;
                    return;
                } else {
                    // dissarm
                    armed = false;
                    coder.init(fs);
                    coder.getRelProbabilityAndAdvance();
                }
            }
            coder.nextBit(bit);

        }

        bool getBit() {
            return exploded ? 1 : coder.getBit();
        }

        Symbol getResult() {
            return exploded ? armedSymbol : coder.getResult();
        }

    };

    int myfilterbits = 0;
    template<template<typename S, typename F> typename Coder, typename FilterLengthStrategy = FilterLengthStrategyOpt, typename Symbol = uint32_t, typename Frequency = float, size_t MAX_FILTER_CODE_LENGTH = 63>
    class FilterCoding {
        Coder<Symbol, Frequency> coder;

    public:

        FilterCoding(size_t cats) : coder(cats) {

        }

        size_t getFilterBits(size_t currentTotal, float p, size_t depth) {
            size_t recommended = FilterLengthStrategy::getFilterBits(p, depth);
            if (recommended + currentTotal <= MAX_FILTER_CODE_LENGTH) {
                // within bounds
                return recommended;
            }
            // allow as much filter bits as possible
            return MAX_FILTER_CODE_LENGTH - currentTotal;
        }

        /*
         * get the filter code
         * a 0 in the filter code is stored in a VLR retrieval structure, a 1 is skipped
         */
        FilterCode encode_once_filter(const std::span<Frequency> &f, Symbol symbol) {
            coder.template init<true>(f, symbol);
            FilterCode res{0, 0};
            size_t depth = 0;
            while (!coder.hasFinished()) {
                float r1=coder.getRelProbabilityAndAdvance();
                uint64_t filterBits = getFilterBits(res.length, r1, depth);
                coder.nextEncodeBit();
                bool r = coder.getBit();
                if (!r) {
                    res.code |= ((uint64_t(1) << filterBits) - 1) << res.length;
                    myfilterbits += filterBits;
                }
                res.length += filterBits;
                depth++;
            }
            return res;
        }


        FilterCode
        encode_once_corrected_code(const std::span<Frequency> &f, Symbol symbol, uint64_t filter_code_data) {
            coder.template init<true>(f, symbol);
            FilterCode res{0, 0};
            size_t totalFilterBitLength = 0;
            size_t depth = 0;
            while (!coder.hasFinished()) {
                uint64_t filterBitLength = getFilterBits(totalFilterBitLength, coder.getRelProbabilityAndAdvance(),
                                                         depth);
                coder.nextEncodeBit();
                totalFilterBitLength += filterBitLength;
                uint64_t filterBits = filter_code_data & ((uint64_t(1) << filterBitLength) - 1);
                filter_code_data >>= filterBitLength;

                if (filterBits == ((uint64_t(1) << filterBitLength) - 1)) {
                    res.code |= (coder.getBit() << res.length);
                    res.length++;
                }

                depth++;
            }
            return res;
        }

        Symbol
        decode_once(const std::span<Frequency> &f, uint64_t corrected_code_data, uint64_t filter_code_data) {
            // the coder has to ensure that at each node the probability of branch 0 is at most 50% (otherwise we would need to swap the filter)
            coder.init(f);
            int depth = 0;
            size_t totalFilterBitLength = 0;
            while (!coder.hasFinished()) {
                float probability = coder.getRelProbabilityAndAdvance();
                assert(probability <= 0.5);
                uint64_t filterBitLength = getFilterBits(totalFilterBitLength, probability, depth);
                totalFilterBitLength += filterBitLength;
                uint64_t filterBits = filter_code_data & ((uint64_t(1) << filterBitLength) - 1);
                filter_code_data >>= filterBitLength;
                if (filterBits == ((uint64_t(1) << filterBitLength) - 1)) {
                    bool nextBit = corrected_code_data & 1;
                    coder.nextBit(nextBit);
                    corrected_code_data >>= 1;
                } else {
                    coder.nextBit(true);
                }
                depth++;
            }
            return coder.getResult();
        }
    };

}