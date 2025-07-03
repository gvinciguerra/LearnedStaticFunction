#pragma once

#include "learnedretrieval/filter_coding.hpp"
#include "learnedretrieval/dataset_reader.hpp"
#include "learnedretrieval/model_wrapper.hpp"

namespace learnedretrieval {

    template<typename Coding>
class FilteredRetrievalStorage {
    using Config = ribbon::RConfig<64, 1, ribbon::ThreshMode::onebit, false, true, false, 0, uint64_t>;
    ribbon::ribbon_filter<4, Config> correctionVLR;
    ribbon::ribbon_filter<4, Config> filterVLR;
    Coding coder;
public:
    double entropy;

    FilteredRetrievalStorage() :coder(0) { }

    template<typename F>
    void build(size_t n, size_t classes_count, F get) {
        rocksdb::StopWatchNano timer(true);
        size_t huffman_bits = 0;
        size_t filter_bits = 0;
        using namespace ribbon;
        IMPORT_RIBBON_CONFIG(Config);

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

        filterVLR = ribbon_filter<4, Config>(0.90625, 42, maxlenfilter);
        filterVLR.AddRange(inputFilter.get(), inputFilter.get() + n, true);
        filterVLR.BackSubst();

        size_t maxlen = 0;
        auto input = std::make_unique<std::pair<Key, ResultRowVLR>[]>(n);
        for (size_t i = 0; i < n; ++i) {
            auto [hash, label, probabilities] = get(i);
            uint64_t filterVal = filterVLR.QueryRetrieval(hash);
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

        correctionVLR =  ribbon_filter<4, Config>(0.90625, 144, maxlen);
        correctionVLR.AddRange(input.get(), input.get() + n);
        correctionVLR.BackSubst();

        auto nanos2 = timer.ElapsedNanos(true);
        std::cout << "Ribbon construction time: " << nanos2 << " ns (" << (nanos2 / static_cast<double>(n))
                  << " ns/item)\n";
        auto totalnanos = nanos + nanos2;
        std::cout << "Total construction time: " << totalnanos << " ns ("
                  << (totalnanos / static_cast<double>(n)) << " ns/item)\n";
        input.reset();

        std::cout << "Max length correction: " << maxlen << "\n";
        std::cout << "Max length filter: " << maxlenfilter << "\n";
        const size_t bytesFilter = filterVLR.Size();
        const size_t bytes = correctionVLR.Size();
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
        uint64_t corrected_code = correctionVLR.QueryRetrieval(hash);
        uint64_t filterCode = filterVLR.QueryRetrieval(hash);
        return {corrected_code, filterCode};
    }

    uint64_t query(uint64_t hash, std::span<float> probabilities) {
        auto [corrected_code, filterCode] = query_storage(hash);
        return coder.decode_once(probabilities, corrected_code, filterCode);
    }

    size_t size_in_bytes() const {
        return filterVLR.Size() + correctionVLR.Size();
    }


    static const std::string get_name() {
         return "Filtered-" + Coding::get_name();
    }
};

template<typename Model, typename Storage>
class LearnedRetrieval {
    XXH3_state_t *state;
    Model &model;
    Storage storage;

public:

    LearnedRetrieval(const learnedretrieval::BinaryDatasetReader &dataset, Model &model) : model(model) {
        state = XXH3_createState();
        assert(state);
        storage = Storage();
        storage.build(
            dataset.size(),
            dataset.classes_count(),
            [&](size_t i) {
                auto example = dataset.get_example(i);
                model.invoke(example);
                return std::make_tuple(hash(i, example), dataset.get_label(i), model.get_probabilities());
            });

        std::cout << "Model size: " << model_bytes() * 8 << " bits\n";
        std::cout << "Total size: " << size_in_bytes() * 8 << " bits\n";
        std::cout << "Total bits/example: " << (size_in_bytes() * 8 / static_cast<double>(dataset.size())) << "\n";
    }

    std::span<float> query_probabilities(std::span<const float> features) {
        model.invoke(features);
        return model.get_probabilities();
    }

    auto query_storage(uint64_t key, std::span<const float> features) {
        return storage.query_storage(hash(key, features));
    }

    uint64_t query(uint64_t key, std::span<const float> features) {
        return storage.query(hash(key, features), query_probabilities(features));
    }

    double get_entropy() const { return storage.entropy; }

    size_t model_bytes() const { return model.model_bytes(); }

    size_t storage_bytes() const { return storage.size_in_bytes(); }

    size_t size_in_bytes() const { return storage.size_in_bytes() + model.model_bytes(); }

private:

    uint64_t hash(uint64_t key, std::span<const float> features) {
        XXH3_64bits_reset_withSeed(state, 500);
        XXH3_64bits_update(state, &key, sizeof(size_t));
        //XXH3_64bits_update(state, features.data(), features.size_bytes());
        return XXH3_64bits_digest(state);
    }
};
}