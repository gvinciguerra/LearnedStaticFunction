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
#include "learnedretrieval/model_wrapper.hpp"

using ModelOutputType = float; // uint8

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <dataset_path> <model_path>" << std::endl;
        return 1;
    }

    auto dataset_path = argv[1];
    auto model_path = argv[2];

    learnedretrieval::BinaryDatasetReader dataset(dataset_path);

    std::cout << "Dataset:" << std::endl
              << "  Examples: " << dataset.size() << std::endl
              << "  Features: " << dataset.features_count() << std::endl
              << "  Classes:  " << dataset.classes_count() << std::endl;

    using namespace ribbon;
    using Config = RConfig<64, 1, ThreshMode::onebit, false, true, false, 0, uint64_t>;
    //using Config = RConfig<64, 1, ThreshMode::onebit, false, true, false, 0, std::span<const float>>;
    IMPORT_RIBBON_CONFIG(Config);
    {
        learnedretrieval::ModelWrapper<ModelOutputType> model(model_path);
        rocksdb::StopWatchNano timer(true);
        size_t huffman_bits = 0;

        learnedretrieval::Huffman<uint64_t, ModelOutputType> coder(dataset.classes_count());
        auto input = std::make_unique<std::pair<Key, ResultRowVLR>[]>(dataset.size());
        //auto input = std::make_unique<std::pair<std::span<const float>, ResultRowVLR>[]>(dataset.size());
        size_t maxlen = 0;
        auto state = XXH3_createState();
        assert(state);
        double entropy = 0;
        size_t numCorrectIsMoreThan50 = 0;
        for (size_t i = 0; i < dataset.size(); ++i) {
            auto example = dataset.get_example(i);
            auto label = dataset.get_label(i);
            auto output = model.invoke(example);
            auto probabilities = model.get_probabilities();
            if (probabilities[label] > 0) {
                entropy += -std::log2(probabilities[label]);
            }
            size_t maxIdx = std::max_element(probabilities.begin(), probabilities.end()) - probabilities.begin();
            if (maxIdx == label && probabilities[label] > 0.5) {
                ++numCorrectIsMoreThan50;
            }
            auto [code, length] = coder.encode_once(output, label);
            XXH3_64bits_reset_withSeed(state, 100);
            XXH3_64bits_update(state, &i, sizeof(size_t));
            XXH3_64bits_update(state, example.data(), example.size_bytes());
            auto hash = XXH3_64bits_digest(state);
            input[i].first = hash;
            //input[i].first = example;
            if (length > maxlen)
                maxlen = length;
            input[i].second = static_cast<uint64_t>(code) | (uint64_t(1) << length);
            huffman_bits += length;
        }
        std::cout << "Entropy: " << (entropy / dataset.size()) << "\n";
        std::cout << "Correct guess has >50% prob: " << (numCorrectIsMoreThan50 / static_cast<double>(dataset.size())) << "\n";
        auto nanos = timer.ElapsedNanos(true);
        std::cout << "Preprocessing time: " << nanos << " ns (" << (nanos / static_cast<double>(dataset.size())) << " ns/item)\n";
        ribbon_filter<4, Config> r(0.90625, 42, maxlen);
        r.AddRange(input.get(), input.get() + dataset.size());
        r.BackSubst();
        auto nanos2 = timer.ElapsedNanos(true);
        std::cout << "Ribbon construction time: " << nanos2 << " ns (" << (nanos2 / static_cast<double>(dataset.size())) << " ns/item)\n";
        auto totalnanos = nanos + nanos2;
        std::cout << "Total construction time: " << totalnanos << " ns (" << (totalnanos / static_cast<double>(dataset.size())) << " ns/item)\n";
        input.reset();
        auto model_bits = model.model_bytes() * 8;

        std::cout << "Max length: " << maxlen << "\n";
        const size_t bytes = r.Size();
        std::cout << "Ribbon size: " << (bytes * 8) << " bits\n";
        std::cout << "Ribbon bits/example: " << ((bytes * 8) / static_cast<double>(dataset.size())) << "\n";
        std::cout << "Model size: " << model_bits << " bits\n";
        std::cout << "Total size: " << (bytes * 8) + model_bits << " bits\n";
        std::cout << "Total bits/example: " << ((bytes * 8 + model_bits) / static_cast<double>(dataset.size())) << "\n";
        std::cout << "Huffman bits: " << huffman_bits << "\n";
        std::cout << "BuRR Overhead (ribbon size/huffman bits): " << ((bytes * 8) / static_cast<double>(huffman_bits)) << "\n";

        volatile auto sum = 0;
        timer.Start();
        for (size_t i = 0; i < dataset.size(); ++i) {
            auto example = dataset.get_example(i);
            auto label = dataset.get_label(i);
            auto output = model.invoke(example);
            sum += output[label];
        }
        nanos = timer.ElapsedNanos(true);
        std::cout << "Model inference time: " << nanos << " ns (" << (nanos / static_cast<double>(dataset.size())) << " ns/query)\n";

        timer.Start();

        bool ok = true;
        for (size_t i = 0; i < dataset.size(); ++i) {
            auto example = dataset.get_example(i);
            auto label = dataset.get_label(i);
            auto output = model.invoke(example);
            XXH3_64bits_reset_withSeed(state, 100);
            XXH3_64bits_update(state, &i, sizeof(size_t));
            XXH3_64bits_update(state, example.data(), example.size_bytes());
            auto hash = XXH3_64bits_digest(state);
            uint64_t val = r.QueryRetrieval(hash);
            //ResultRowVLR val = r.QueryRetrieval(example);
            size_t bit_offset = 0;
            uint64_t res = coder.decode_once(output, &val, bit_offset);
            bool found = res == label;
            assert(found);
            ok &= found;
        }
        XXH3_freeState(state);
        if (!ok)
            std::cerr << "FAILED\n";
        nanos = timer.ElapsedNanos(true);
        std::cout << "Query time: " << nanos << " ns (" << (nanos / static_cast<double>(dataset.size())) << " ns/query)\n";

        #if 0
        auto repetitions = 10000000;
        volatile auto total = 0;

        auto start = std::chrono::high_resolution_clock::now();
        for (auto i = 0; i < repetitions; ++i) {
            auto example = dataset.get_example(i % dataset.size());
            auto output = model.invoke(example);
            total += output[i % output.size()];
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        std::cout << "  Inference ns/query:    " << duration.count() / repetitions << std::endl;

        auto start2 = std::chrono::high_resolution_clock::now();
        for (auto i = 0; i < repetitions; ++i) {
            auto example = dataset.get_example(i % dataset.size());
            auto label = dataset.get_label(i % dataset.size());
            auto output = model.invoke(example);
            auto [code, length] = coder.encode_once(output, label);
            total += length;
        }
        auto end2 = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2);
        std::cout << "  Overall ns/query:      " << duration2.count() / repetitions << std::endl;
        #endif
    }

    return 0;
}
