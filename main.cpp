#include <iostream>
#include <chrono>

#include "learnedretrieval/huffman.hpp"
#include "learnedretrieval/model_wrapper.hpp"
#include "learnedretrieval/tree_huffman.hpp"
#include "learnedretrieval/dataset_reader.hpp"

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
        std::vector<uint32_t> frequencies(dataset.classes_count());
        for (int i = 0; i < dataset.size(); ++i)
            ++frequencies[dataset.get_label(i)];

        // Compute total encoding size
        learnedretrieval::Huffman<uint16_t, uint32_t> huffman(frequencies);
        size_t total_bits = 0;
        for (int c = 0; c < dataset.classes_count(); ++c) {
            auto [code, length] = huffman.encode(c);
            total_bits += frequencies[c] * length;
        }
        std::cout << "Huffman on values bits/value: " << double(total_bits) / dataset.size() << std::endl;

        // Encode in a temporary buffer
        size_t output_words = (total_bits + 63) / 64;
        std::vector<uint64_t> encoded_data(output_words);
        size_t bit_offset = 0;
        for (auto i = 0; i < dataset.size(); ++i) {
            auto label = dataset.get_label(i);
            huffman.encode_write(label, encoded_data.data(), bit_offset);
        }

        // Decode and check
        bit_offset = 0;
        for (int i = 0; i < dataset.size(); ++i) {
            auto label = dataset.get_label(i);
            auto decoded_label = huffman.decode(encoded_data.data(), bit_offset);
            if (label != decoded_label)
                throw std::runtime_error("Decoded label at " + std::to_string(i) + " mismatch");
        }
    }

    // Learned retrieval coding of labels
    {
        learnedretrieval::ModelWrapper model(model_path);
        std::vector<uint32_t> symbols(dataset.classes_count());
        size_t huffman_bits = 0;

        for (int i = 0; i < dataset.size(); ++i) {
            auto example = dataset.get_example(i);
            auto label = dataset.get_label(i);
            auto output = model.invoke(example);
            auto [code, length] = learnedretrieval::Huffman<uint32_t, float>::encode_once(symbols, output, label);
            huffman_bits += length;
        }
        auto model_bits = model.model_bytes() * 8;
        std::cout << "Learned retrieval:" << std::endl
                  << "  Model bits/value:      " << double(model_bits) / dataset.size() << std::endl
                  << "  Retrieval bits/value:  " << double(huffman_bits) / dataset.size() << std::endl
                  << "  Overall bits/value:    " << double(huffman_bits + model_bits) / dataset.size() << std::endl;

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
            auto [code, length] = learnedretrieval::Huffman<>::encode_once(symbols, output, label);
            total += length;
        }
        auto end2 = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2);
        std::cout << "  Overall ns/query:      " << duration2.count() / repetitions << std::endl;
    }

    return 0;
}
