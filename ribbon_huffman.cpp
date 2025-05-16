#include <atomic>
#include <cstdlib>
#include <numeric>
#include <thread>
#include <vector>
#include <iostream>
#include <chrono>

#include "ribbon.hpp"
#include "serialization.hpp"
#include "rocksdb/stop_watch.h"

#include <tlx/cmdline_parser.hpp>
#include <tlx/logger.hpp>

#include "learnedretrieval/huffman.hpp"
#include "learnedretrieval/dataset_reader.hpp"

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <dataset_path>" << std::endl;
        return 1;
    }

    auto dataset_path = argv[1];

    learnedretrieval::BinaryDatasetReader dataset(dataset_path);

    std::cout << "Dataset:" << std::endl
              << "  Examples: " << dataset.size() << std::endl
              << "  Features: " << dataset.features_count() << std::endl
              << "  Classes:  " << dataset.classes_count() << std::endl;

    using namespace ribbon;
    // Plain Huffman coding of labels
    {
        using Config = RConfig<64, 1, ThreshMode::onebit, false, true, false, 0, uint64_t>;
        //using Config = RConfig<64, 1, ThreshMode::onebit, false, true, false, 0, std::span<const float>>;
        IMPORT_RIBBON_CONFIG(Config);
        rocksdb::StopWatchNano timer(true);
        std::vector<uint32_t> frequencies(dataset.classes_count());
        for (size_t i = 0; i < dataset.size(); ++i)
            ++frequencies[dataset.get_label(i)];

        /*
        double entropy = 0;
        for (auto freq : frequencies) {
            double p = freq / static_cast<double>(dataset.size());
            entropy -=  p * std::log2(p);
        }
        std::cout << "Entropy: " << entropy << "\n";
        */

        // Compute total encoding size
        learnedretrieval::Huffman<uint16_t, uint32_t> huffman(frequencies);
        auto input = std::make_unique<std::pair<Key, ResultRowVLR>[]>(dataset.size());
        //auto input = std::make_unique<std::pair<std::span<const float>, ResultRowVLR>[]>(dataset.size());

        size_t maxlen = 0;
        auto state = XXH3_createState();
        assert(state);
        size_t huffman_bits = 0;
        for (size_t i = 0; i < dataset.size(); ++i) {
            auto label = dataset.get_label(i);
            auto example = dataset.get_example(i);
            auto [code, length] = huffman.encode(label);
            //input[i].first = XXH3_64bits_withSeed(example.data(), example.size_bytes(), 100);
            XXH3_64bits_reset_withSeed(state, 100);
            XXH3_64bits_update(state, &i, sizeof(size_t));
            XXH3_64bits_update(state, example.data(), example.size_bytes());
            input[i].first = XXH3_64bits_digest(state);
            //input[i].first = example;
            if (length > maxlen)
                maxlen = length;
            input[i].second = code | (uint16_t(1) << length);
            huffman_bits += length;
        }
        auto nanos = timer.ElapsedNanos(true);
        std::cout << "Preprocessing time: " << nanos << " ns (" << (nanos / static_cast<double>(dataset.size())) << " ns/item)\n";
        ribbon_filter<4, Config> r(0.90625, 42, maxlen);
        //std::cout << "Allocation took " << timer.ElapsedNanos(true) / 1e6 << "ms\n";
        r.AddRange(input.get(), input.get() + dataset.size());
        //std::cout << "Insertion took " << timer.ElapsedNanos(true) / 1e6 << "ms in total\n";
        r.BackSubst();
        //std::cout << "Backsubstitution took " << timer.ElapsedNanos(true) / 1e6 << "ms in total\n";
        auto nanos2 = timer.ElapsedNanos(true);
        std::cout << "Ribbon construction time: " << nanos2 << " ns (" << (nanos2 / static_cast<double>(dataset.size())) << " ns/item)\n";
        auto totalnanos = nanos + nanos2;
        std::cout << "Total construction time: " << totalnanos << " ns (" << (totalnanos / static_cast<double>(dataset.size())) << " ns/item)\n";
        input.reset();
        const size_t bytes = r.Size() + frequencies.size() * sizeof(uint32_t);
        std::cout << "Size: " << (bytes * 8) << " bits (" << ((bytes * 8) / static_cast<double>(dataset.size())) << " bits/item)\n";
        std::cout << "Total huffman bits: " << huffman_bits << "\n";
        std::cout << "Max huffman length: " << maxlen << "\n";

        timer.Start();

        bool ok = true;
        for (size_t i = 0; i < dataset.size(); ++i) {
            auto example = dataset.get_example(i);
            auto label = dataset.get_label(i);
            XXH3_64bits_reset_withSeed(state, 100);
            XXH3_64bits_update(state, &i, sizeof(size_t));
            XXH3_64bits_update(state, example.data(), example.size_bytes());
            auto hash = XXH3_64bits_digest(state);
            uint64_t val = r.QueryRetrieval(hash);
            //ResultRowVLR val = r.QueryRetrieval(example);
            size_t bit_offset = 0;
            uint64_t res = huffman.decode(&val, bit_offset);
            bool found = res == label;
            assert(found);
            ok &= found;
        }
        XXH3_freeState(state);
        if (!ok)
            std::cerr << "FAILED\n";
        nanos = timer.ElapsedNanos(true);
        std::cout << "Query time: " << nanos << " ns (" << (nanos / static_cast<double>(dataset.size())) << " ns/query)\n";
    }

    return 0;
}
