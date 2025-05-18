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

namespace ribbon {
    struct Config : public ribbon::RConfig<64, 1, ThreshMode::onebit, false, true, false, 0, uint64_t> {
        static constexpr bool kUseVLR = false;
    };
}

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
    IMPORT_RIBBON_CONFIG(Config);
    {
        learnedretrieval::ModelWrapper<ModelOutputType> model(model_path);
        rocksdb::StopWatchNano timer(true);
        learnedretrieval::Huffman<uint64_t, ModelOutputType> coder(dataset.classes_count());

        size_t huffman_bits = 0;
        size_t maxlen = 0;
        // Since the encoding is so expensive here, the std::vector version with reallocation is much
        // faster than calculating the number of bits beforehand.
        #if 0
        // I guess it would also be possible to ust std::vector instead of first calculating totalbits
        for (size_t i = 0; i < dataset.size(); ++i) {
            auto example = dataset.get_example(i);
            auto label = dataset.get_label(i);
            auto output = model.invoke(example);
            auto [code, length] = coder.encode_once(output, label);
            if (length > maxlen)
                maxlen = length;
            huffman_bits += length;
        }
        #endif

        //auto input = std::make_unique<std::pair<Key, ResultRow>[]>(huffman_bits);
        std::vector<std::pair<Key, ResultRow>> input;
        // it will actually be a lot bigger because all bits are contained individually
        input.reserve(dataset.size());
        auto state = XXH3_createState();
        assert(state);
        size_t cur = 0;
        for (size_t i = 0; i < dataset.size(); ++i) {
            auto example = dataset.get_example(i);
            auto label = dataset.get_label(i);
            auto output = model.invoke(example);
            auto [code, length] = coder.encode_once(output, label);

            for (unsigned char j = 0; j < length; ++j) {
                XXH3_64bits_reset_withSeed(state, 100);
                XXH3_64bits_update(state, &i, sizeof(size_t));
                XXH3_64bits_update(state, &j, sizeof(unsigned char));
                XXH3_64bits_update(state, example.data(), example.size_bytes());
                /*input[cur].first = XXH3_64bits_digest(state);
                input[cur].second = code & 1;*/
                input.emplace_back(XXH3_64bits_digest(state), code & 1);
                code >>= 1;
                ++cur;
            }
            if (length > maxlen)
                maxlen = length;
            huffman_bits += length;
        }
        auto nanos = timer.ElapsedNanos(true);
        std::cout << "Preprocessing time: " << nanos << " ns (" << (nanos / static_cast<double>(dataset.size())) << " ns/item)\n";
        ribbon_filter<4, Config> r(static_cast<size_t>(huffman_bits * 0.90625), 0.90625, 42);
        //r.AddRange(input.get(), input.get() + huffman_bits);
        r.AddRange(input.begin(), input.end());
        r.BackSubst();
        auto nanos2 = timer.ElapsedNanos(true);
        std::cout << "Ribbon construction time: " << nanos2 << " ns (" << (nanos2 / static_cast<double>(dataset.size())) << " ns/item)\n";
        auto totalnanos = nanos + nanos2;
        std::cout << "Total construction time: " << totalnanos << " ns (" << (totalnanos / static_cast<double>(dataset.size())) << " ns/item)\n";
        //input.reset();
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

        timer.Start();

        bool ok = true;
        for (size_t i = 0; i < dataset.size(); ++i) {
            auto example = dataset.get_example(i);
            auto label = dataset.get_label(i);
            auto output = model.invoke(example);
            uint64_t code = 0;
            for (unsigned char j = maxlen; j-- > 0;) {
                XXH3_64bits_reset_withSeed(state, 100);
                XXH3_64bits_update(state, &i, sizeof(size_t));
                XXH3_64bits_update(state, &j, sizeof(unsigned char));
                XXH3_64bits_update(state, example.data(), example.size_bytes());
                auto hash = XXH3_64bits_digest(state);
                auto val = r.QueryRetrieval(hash);
                code <<= 1;
                code |= (val & 1);
            }
            size_t bit_offset = 0;
            uint64_t res = coder.decode_once(output, &code, bit_offset);
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
