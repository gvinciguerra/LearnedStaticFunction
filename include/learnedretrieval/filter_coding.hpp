#pragma once

#pragma once

#include <algorithm>
#include <cassert>
#include <vector>
#include <ranges>
#include <numeric>
#include <queue>
#include <functional>
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

    class FilterLengthStrategyOpt {
        static constexpr int MAX_FILTER_BITS = 5;
        static constexpr double PROBABILITY_THRESHOLDS[MAX_FILTER_BITS] = {0.333333, 0.2, 0.111111, 0.0588235, 0.030303, /*0.0153846,
                                                               0.00775194, 0.00389105, 0.00194932, 0.00097561,*/};
    public:
        static uint64_t getFilterBits(float probability, size_t level) {
            //if(level>0)return 0;
            size_t bits = 0;
            while (bits < MAX_FILTER_BITS && PROBABILITY_THRESHOLDS[bits] > probability)
                bits++;
            return bits;
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

        class Compare {
        public:
            bool operator()(Node a, Node b) {
                return a.p > b.p;
            }
        };


        FilterHuffmanCoder(const std::span<Frequency> &f) {
            std::priority_queue<Node, std::vector<Node>, Compare> nodes;
            for (size_t i = 0; i < f.size(); ++i) {
                Node n{i, 0, 0, 0, 0, f[i], 0, i, true};
                nodes.push(n);
                tree.push_back(n);
            }
            while (nodes.size() > 1) {
                Node a = nodes.top();
                nodes.pop();
                a.bitRelParent = 0;
                a.parent = tree.size();
                tree[a.index] = a;
                Node b = nodes.top();
                nodes.pop();
                b.bitRelParent = 1;
                b.parent = tree.size();
                tree[b.index] = b;
                Node parent{0, a.index, b.index, 0, 0, a.p + b.p, a.p / (a.p + b.p), tree.size(), false};
                /*std::cout << (a.p + b.p) << " " << parent.index << " " << a.index << " " << b.index << " " << a.p << " "
                          << b.p << " " << parent.relP << std::endl;*/
                if (std::isnan(parent.relP)) {
                    //std::cerr<<"p nan"<<std::endl;
                    parent.relP = 0.5;
                }
                tree.push_back(parent);
                nodes.push(parent);
            }
            root = nodes.top();
            currentDecodingNode = root;
        }

        std::vector<std::pair<float, bool>> getProbabilities(Symbol s) {
            std::vector<std::pair<float, bool>> res;
            Node current = tree[s];
            while (current.index != root.index) {
                bool b = current.bitRelParent;
                current = tree[current.parent];
                res.push_back({current.relP, b});
            }
            return res;
        }

        float getRelProbability() {
            return currentDecodingNode.relP;
        }

        bool hasFinished() {
            return currentDecodingNode.leaf;
        }

        void nextBit(bool bit) {
            currentDecodingNode = tree[bit ? currentDecodingNode.n2 : currentDecodingNode.n1];
        }

        Symbol getResult() {
            return currentDecodingNode.s;
        }
    };

    template<typename Coder, typename FilterLengthStrategy = FilterLengthStrategyOpt, typename Symbol = uint32_t, typename Frequency = float>
    class FilterCoding {
    public:
        /*
         * get the filter code
         * a 0 in the filter code is stored in a VLR retrieval structure, a 1 is skipped
         */
        static FilterCode encode_once_filter(const std::span<Frequency> &f, Symbol symbol) {
            Coder coder(f);
            std::vector<std::pair<float, bool>> probs = coder.getProbabilities(symbol);
            FilterCode res{0, 0};
            for (size_t i = 0; i < probs.size(); ++i) {
                auto [p, b] = probs[i];
                uint64_t filterBits = FilterLengthStrategy::getFilterBits(p, probs.size() - i - 1);
                res.length += filterBits;
                res.code <<= filterBits;
                if (!b) {
                    res.code |= ((1 << filterBits) - 1);
                }
            }
            return res;
        }

        static FilterCode
        encode_once_corrected_code(const std::span<Frequency> &f, Symbol symbol, uint64_t filter_code_data) {
            Coder coder(f);
            std::vector<std::pair<float, bool>> probs = coder.getProbabilities(symbol);

            FilterCode res{0, 0};
            for (size_t i = 0; i < probs.size(); ++i) {
                auto [p, b] = probs[probs.size() - i - 1];

                uint64_t filterBits = FilterLengthStrategy::getFilterBits(p, i);
                uint64_t nodeFilterBits = filter_code_data & ((1 << filterBits) - 1);
                filter_code_data >>= filterBits;

                if (nodeFilterBits == ((1 << filterBits) - 1)) {
                    res.code |= (b << res.length);
                    res.length++;
                }

            }
            return res;
        }

        static Symbol decode_once(const std::span<Frequency> &f, uint64_t corrected_code_data, uint64_t filter_code_data) {
            // the coder has to ensure that at each node the probability of branch 0 is at most 50% (otherwise we would need to swap the filter)
            Coder coder(f);
            int depth = 0;
            while (!coder.hasFinished()) {
                float probability = coder.getRelProbability();
                assert(probability <= 0.5);
                uint64_t filterBitLength = FilterLengthStrategy::getFilterBits(probability, depth);
                uint64_t filterBits = filter_code_data & ((1 << filterBitLength) - 1);
                filter_code_data >>= filterBitLength;
                if (filterBits == ((1 << filterBitLength) - 1)) {
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