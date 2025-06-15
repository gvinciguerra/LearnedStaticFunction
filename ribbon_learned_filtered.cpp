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

#include "learnedretrieval/filter_coding.hpp"
#include "learnedretrieval/dataset_reader.hpp"
#include "learnedretrieval/model_wrapper.hpp"

using ModelOutputType = float; // uint8_t

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
    using FilerConfig = Config;
    IMPORT_RIBBON_CONFIG(Config);
    {
        learnedretrieval::ModelWrapper<ModelOutputType> model(model_path);
        rocksdb::StopWatchNano timer(true);
        size_t huffman_bits = 0;
        size_t filter_bits = 0;

        using coder =  learnedretrieval::FilterCoding<learnedretrieval::FilterFanoCoder<>>;
        auto state = XXH3_createState();
        assert(state);
        size_t maxlenfilter = 0;
        auto inputFilter = std::make_unique<std::pair<Key, ResultRowVLR>[]>(dataset.size());
        for (size_t i = 0; i < dataset.size(); ++i) {

            auto example = dataset.get_example(i);
            auto label = dataset.get_label(i);
            auto output = model.invoke(example);
            auto probabilities = model.get_probabilities();
            auto [code, filterLength] = coder::encode_once_filter(probabilities, label);
            XXH3_64bits_reset_withSeed(state, 500);
            XXH3_64bits_update(state, &i, sizeof(size_t));
            XXH3_64bits_update(state, example.data(), example.size_bytes());
            auto hash = XXH3_64bits_digest(state);
            inputFilter[i].first = hash;
            if (filterLength > maxlenfilter)
                maxlenfilter = filterLength;
            inputFilter[i].second = static_cast<uint64_t>(code) | (uint64_t(1) << filterLength);
            filter_bits += filterLength;
        }


        ribbon_filter<4, Config> filter(0.90625, 42, maxlenfilter);
        filter.AddRange(inputFilter.get(), inputFilter.get() + dataset.size(), true);
        filter.BackSubst();

        size_t maxlen = 0;
        auto input = std::make_unique<std::pair<Key, ResultRowVLR>[]>(dataset.size());
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
            auto hash = inputFilter[i].first;

            // get the output filter
            uint64_t filterVal = filter.QueryRetrieval(hash);

            size_t bit_offset = 0;
            auto [code, length] = coder::encode_once_corrected_code(probabilities, label, filterVal);
            input[i].first = hash;
            if (length > maxlen)
                maxlen = length;
            input[i].second = static_cast<uint64_t>(code) | (uint64_t(1) << length);
            huffman_bits += length;
        }

        std::cout << "Entropy: " << (entropy / dataset.size()) << "\n";
        std::cout << "Correct guess has >50% prob: " << (numCorrectIsMoreThan50 / static_cast<double>(dataset.size())) << "\n";
        auto nanos = timer.ElapsedNanos(true);
        std::cout << "Preprocessing time (including filter): " << nanos << " ns (" << (nanos / static_cast<double>(dataset.size())) << " ns/item)\n";
        ribbon_filter<4, Config> r(0.90625, 144, maxlen);
        r.AddRange(input.get(), input.get() + dataset.size());
        r.BackSubst();
        auto nanos2 = timer.ElapsedNanos(true);
        std::cout << "Ribbon construction time: " << nanos2 << " ns (" << (nanos2 / static_cast<double>(dataset.size())) << " ns/item)\n";
        auto totalnanos = nanos + nanos2;
        std::cout << "Total construction time: " << totalnanos << " ns (" << (totalnanos / static_cast<double>(dataset.size())) << " ns/item)\n";
        input.reset();
        auto model_bits = model.model_bytes() * 8;

        std::cout << "Max length correction: " << maxlen << "\n";
        std::cout << "Max length filter: " << maxlenfilter << "\n";
        const size_t bytesFilter = filter.Size();
        const size_t bytes = r.Size();
        const size_t bytesTotal = bytes + bytesFilter;
        std::cout << "Ribbon size: " << (bytes * 8) << " bits\n";
        std::cout << "Filter size: " << (bytesFilter * 8) << " bits\n";
        std::cout << "Ribbon+Filter size: " << (bytesTotal * 8) << " bits\n";
        std::cout << "Ribbon bits/example: " << ((bytes * 8) / static_cast<double>(dataset.size())) << "\n";
        std::cout << "Filter bits/example: " << ((bytesFilter * 8) / static_cast<double>(dataset.size())) << "\n";
        std::cout << "Ribbon+Filter bits/example: " << ((bytesTotal * 8) / static_cast<double>(dataset.size())) << "\n";
        std::cout << "Model size: " << model_bits << " bits\n";
        std::cout << "Total size: " << (bytesTotal * 8) + model_bits << " bits\n";
        std::cout << "Total bits/example: " << ((bytesTotal * 8 + model_bits) / static_cast<double>(dataset.size())) << "\n";
        std::cout << "Huffman bits: " << huffman_bits << "\n";
        //std::cout << "BuRR Overhead (ribbon size/huffman bits): " << ((bytes * 8) / static_cast<double>(huffman_bits)) << "\n";

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
            XXH3_64bits_reset_withSeed(state, 500);
            XXH3_64bits_update(state, &i, sizeof(size_t));
            XXH3_64bits_update(state, example.data(), example.size_bytes());
            auto hash = XXH3_64bits_digest(state);
            uint64_t corrected_code = r.QueryRetrieval(hash);
            uint64_t filterCode = filter.QueryRetrieval(hash);
            uint64_t res = coder::decode_once(model.get_probabilities(), corrected_code, filterCode);
            bool found = res == label;
            assert(found);
            ok &= found;
        }
        XXH3_freeState(state);
        if (!ok)
            std::cerr << "FAILED\n";
        nanos = timer.ElapsedNanos(true);
        std::cout << "Query time: " << nanos << " ns (" << (nanos / static_cast<double>(dataset.size())) << " ns/query)\n";
std::cout<<learnedretrieval::myfilterbits<<" "<<learnedretrieval::maxFilterCnt<<std::endl;

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
