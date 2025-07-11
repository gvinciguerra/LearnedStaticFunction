#pragma once

#include "learnedretrieval/filter_coding.hpp"
#include "learnedretrieval/dataset_reader.hpp"
#include "learnedretrieval/model_wrapper.hpp"

namespace lsf {


    constexpr float slotsPerItem = 0.95;
    struct BuRRConfig
            : public ribbon::RConfig<64, 1, ribbon::ThreshMode::onebit, false, true, false, 0, uint64_t> {
        static constexpr bool kUseVLR = true;
    };


    template<typename Coding>
    class FilteredLSFStorage {
        ribbon::ribbon_filter<4, BuRRConfig> correctionVLSF;
        ribbon::ribbon_filter<4, BuRRConfig> filterVLSF;
        Coding coder;
    public:

        FilteredLSFStorage() : coder(0) {}

        template<typename F>
        void build(size_t n, size_t classes_count, F get) {
            rocksdb::StopWatchNano timer(true);
            size_t huffman_bits = 0;
            size_t filter_bits = 0;
            using namespace ribbon;
            IMPORT_RIBBON_CONFIG(BuRRConfig);

            coder = Coding(classes_count);
            size_t maxlenfilter = 0;
            auto inputFilter = std::make_unique<std::pair<Key, ResultRowVLR>[]>(n);
            for (size_t i = 0; i < n; ++i) {
                auto [hash, label, probabilities] = get(i);
                auto [code, filterLength] = coder.encode_once_filter(probabilities, label);
                inputFilter[i].first = hash;
                if (filterLength > maxlenfilter)
                    maxlenfilter = filterLength;
                inputFilter[i].second = static_cast<uint64_t>(code) | (uint64_t(1) << filterLength);
                filter_bits += filterLength;
            }

            filterVLSF = ribbon_filter<4, BuRRConfig>(slotsPerItem, 42, maxlenfilter);
            filterVLSF.AddRange(inputFilter.get(), inputFilter.get() + n, true);
            filterVLSF.BackSubst();

            size_t maxlen = 0;
            auto input = std::make_unique<std::pair<Key, ResultRowVLR>[]>(n);
            for (size_t i = 0; i < n; ++i) {
                auto [hash, label, probabilities] = get(i);
                uint64_t filterVal = filterVLSF.QueryRetrieval(hash);
                auto [code, length] = coder.encode_once_corrected_code(probabilities, label, filterVal);
                input[i].first = hash;
                if (length > maxlen)
                    maxlen = length;
                input[i].second = static_cast<uint64_t>(code) | (uint64_t(1) << length);
                huffman_bits += length;
            }

            auto nanos = timer.ElapsedNanos(true);
            std::cout << "Preprocessing time (including filter): " << nanos << " ns ("
                      << (nanos / static_cast<double>(n)) << " ns/item)\n";

            correctionVLSF = ribbon_filter<4, BuRRConfig>(slotsPerItem, 42, maxlen);
            correctionVLSF.AddRange(input.get(), input.get() + n);
            correctionVLSF.BackSubst();

            auto nanos2 = timer.ElapsedNanos(true);
            std::cout << "Ribbon construction time: " << nanos2 << " ns (" << (nanos2 / static_cast<double>(n))
                      << " ns/item)\n";
            auto totalnanos = nanos + nanos2;
            std::cout << "Total construction time: " << totalnanos << " ns ("
                      << (totalnanos / static_cast<double>(n)) << " ns/item)\n";
            input.reset();

            std::cout << "Max length correction: " << maxlen << "\n";
            std::cout << "Max length filter: " << maxlenfilter << "\n";
            const size_t bytesFilter = filterVLSF.Size();
            const size_t bytes = correctionVLSF.Size();
            const size_t bytesTotal = bytes + bytesFilter;
            std::cout << "Ribbon size: " << (bytes * 8) << " bits\n";
            std::cout << "Filter size: " << (bytesFilter * 8) << " bits\n";
            std::cout << "Ribbon+Filter size: " << (bytesTotal * 8) << " bits\n";
            std::cout << "Ribbon bits/example: " << ((bytes * 8) / static_cast<double>(n)) << "\n";
            std::cout << "Filter bits/example: " << ((bytesFilter * 8) / static_cast<double>(n)) << "\n";
            std::cout << "Ribbon+Filter bits/example: " << ((bytesTotal * 8) / static_cast<double>(n)) << "\n";
            std::cout << "Huffman bits: " << huffman_bits << "\n";
        }

        std::pair<uint64_t, uint64_t> query_storage(uint64_t hash) {
            uint64_t corrected_code = correctionVLSF.QueryRetrieval(hash);
            uint64_t filterCode = filterVLSF.QueryRetrieval(hash);
            return {corrected_code, filterCode};
        }

        uint64_t query(uint64_t hash, std::span<float> probabilities) {
            auto [corrected_code, filterCode] = query_storage(hash);
            return coder.decode_once(probabilities, corrected_code, filterCode);
        }

        size_t size_in_bytes() const {
            return filterVLSF.Size() + correctionVLSF.Size();
        }


        static const std::string get_name() {
            return "Filtered-" + Coding::get_name();
        }
    };

    template<typename Model, typename Storage>
    class LearnedStaticFunction {
        XXH3_state_t *state;
        Model &model;
        Storage storage;

    public:

        LearnedStaticFunction(const lsf::BinaryDatasetReader &dataset, Model &model) : model(model) {
            state = XXH3_createState();
            assert(state);
            storage = Storage();
            storage.build(
                    dataset.size(),
                    dataset.classes_count(),
                    [&](size_t i) {
                        auto example = dataset.get_example(i);
                        return std::make_tuple(hash(i, example), dataset.get_label(i), model.invoke(example));
                    });

            std::cout << "Model size: " << model_bytes() * 8 << " bits\n";
            std::cout << "Total size: " << size_in_bytes() * 8 << " bits\n";
            std::cout << "Total bits/example: " << (size_in_bytes() * 8 / static_cast<double>(dataset.size())) << "\n";
        }

        std::span<float> query_probabilities(std::span<const float> features) {
            return model.invoke(features);
        }

        auto query_storage(uint64_t key, std::span<const float> features) {
            return storage.query_storage(hash(key, features));
        }

        uint64_t query(uint64_t key, std::span<const float> features) {
            return storage.query(hash(key, features), query_probabilities(features));
        }

        size_t model_bytes() const { return model.model_bytes(); }

        size_t storage_bytes() const { return storage.size_in_bytes(); }

        size_t size_in_bytes() const { return storage.size_in_bytes() + model.model_bytes(); }

    private:

        uint64_t hash(uint64_t key, std::span<const float> features) {
            XXH3_64bits_reset(state);
            XXH3_64bits_update(state, &key, sizeof(size_t));
            //XXH3_64bits_update(state, features.data(), features.size_bytes());
            return XXH3_64bits_digest(state);
        }
    };
}