#include <iostream>
#include <chrono>

#include "learnedretrieval/huffman.hpp"
#include "learnedretrieval/model_wrapper.hpp"
#include "learnedretrieval/shannon.hpp"
#include "learnedretrieval/dataset_reader.hpp"

template<typename Coder, typename Frequencies>
void value_encoding(learnedretrieval::BinaryDatasetReader &dataset, Frequencies &frequencies) {
    std::vector<float> normalized_frequencies(frequencies.size());
    for (int i = 0; i < frequencies.size(); ++i)
        normalized_frequencies[i] = float(frequencies[i]) / dataset.size();

    Coder coder(normalized_frequencies);
    size_t total_bits = 0;
    for (int c = 0; c < dataset.classes_count(); ++c) {
        auto [code, length] = coder.encode(c);
        total_bits += frequencies[c] * length;
    }
    std::cout << "  bits/value: " << double(total_bits) / dataset.size() << std::endl;

    // Encode in a temporary buffer
    auto t0 = std::chrono::high_resolution_clock::now();
    size_t output_words = (total_bits + 63) / 64;
    std::vector<uint64_t> encoded_data(output_words);
    size_t bit_offset = 0;
    for (auto i = 0; i < dataset.size(); ++i) {
        auto label = dataset.get_label(i);
        coder.encode_write(label, encoded_data.data(), bit_offset);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0);
    std::cout << "  Encoding time: " << duration.count() / dataset.size() << " ns/value" << std::endl;

    // Decode and check
    t0 = std::chrono::high_resolution_clock::now();
    bit_offset = 0;
    for (int i = 0; i < dataset.size(); ++i) {
        auto label = dataset.get_label(i);
        auto decoded_label = coder.decode(encoded_data.data(), bit_offset);
        if (label != decoded_label)
            throw std::runtime_error("Decoded label at " + std::to_string(i) + " mismatch");
    }
    t1 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0);
    std::cout << "  Decoding time: " << duration.count() / dataset.size() << " ns/value" << std::endl;
}

template<typename Coder>
void learned_value_encoding(learnedretrieval::BinaryDatasetReader &dataset, learnedretrieval::ModelWrapper &model) {
    std::vector<uint32_t> symbols(dataset.classes_count());
    size_t huffman_bits = 0;

    std::vector<float> tmp(dataset.classes_count());
    Coder coder(tmp);

    for (int i = 0; i < dataset.size(); ++i) {
        auto example = dataset.get_example(i);
        auto label = dataset.get_label(i);
        auto output = model.invoke(example);
        coder.reset(output);
        auto [code, length] = coder.encode(label);
        huffman_bits += length;
    }
    auto model_bits = model.model_bytes() * 8;
    std::cout << "  Model bits/value:      " << double(model_bits) / dataset.size() << std::endl
              << "  Retrieval bits/value:  " << double(huffman_bits) / dataset.size() << std::endl
              << "  Overall bits/value:    " << double(huffman_bits + model_bits) / dataset.size() << std::endl;

    auto repetitions = 1000000;
    volatile auto total = 0;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (auto i = 0; i < repetitions; ++i) {
        auto example = dataset.get_example(i % dataset.size());
        auto output = model.invoke(example);
        total += output[i % output.size()];
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0);
    std::cout << "  Inference ns/query:    " << duration.count() / repetitions << std::endl;

    std::vector<uint64_t> encoded_data((repetitions * std::ceil(std::log2(dataset.classes_count()))) / 64);
    size_t bit_offset = 0;

    t0 = std::chrono::high_resolution_clock::now();
    for (auto i = 0; i < repetitions; ++i) {
        auto example = dataset.get_example(i % dataset.size());
        auto label = dataset.get_label(i % dataset.size());
        auto output = model.invoke(example);
        coder.reset(output);
        coder.encode_write(label, encoded_data.data(), bit_offset);
        total += bit_offset;
    }
    t1 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0);
    std::cout << "  Overall enc ns/query:  " << duration.count() / repetitions << std::endl;

    bit_offset = 0;
    t0 = std::chrono::high_resolution_clock::now();
    for (auto i = 0; i < repetitions; ++i) {
        auto example = dataset.get_example(i % dataset.size());
        auto label = dataset.get_label(i % dataset.size());
        auto output = model.invoke(example);
        coder.reset(output);
        auto decoded_label = coder.decode(encoded_data.data(), bit_offset);
        total += bit_offset;
        if (label != decoded_label)
            throw std::runtime_error("Decoded label at " + std::to_string(i) + " mismatch");
    }
    t1 = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0);
    std::cout << "  Overall dec ns/query:  " << duration.count() / repetitions << std::endl;
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

    // Plain Huffman coding of labels
    {
        std::vector<float> frequencies(dataset.classes_count());
        for (int i = 0; i < dataset.size(); ++i)
            ++frequencies[dataset.get_label(i)];

        double entropy = 0;
        for (int i = 0; i < dataset.classes_count(); ++i) {
            auto f = frequencies[i] / dataset.size();
            if (f > 0)
                entropy -= f * std::log2(f);
        }

        std::cout << "Entropy: " << entropy << std::endl;
        std::cout << "Plain Huffman:" << std::endl;
        value_encoding<learnedretrieval::Huffman<uint16_t, float>, decltype(frequencies)>(dataset, frequencies);
        std::cout << "Plain Shannon:" << std::endl;
        value_encoding<learnedretrieval::Shannon<uint16_t, float>, decltype(frequencies)>(dataset, frequencies);
    }

    // Learned retrieval coding of labels
    {
        learnedretrieval::ModelWrapper model(model_path);
        std::cout << "Learned retrieval (Huffman):" << std::endl;
        learned_value_encoding<learnedretrieval::Huffman<uint32_t, float>>(dataset, model);
        std::cout << "Learned retrieval (Shannon):" << std::endl;
        learned_value_encoding<learnedretrieval::Shannon<uint32_t, float>>(dataset, model);
    }

    return 0;
}
