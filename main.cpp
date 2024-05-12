#include <iostream>

#include "learnedretrieval/dataset_reader.hpp"
#include "learnedretrieval/huffman.hpp"

int main(int argc, char *argv[]) {
    std::vector<std::pair<char, float>> freqs_test{{'a', 5.}, {'b', 2.}, {'c', 1.}, {'d', 1.}, {'r', 2.}};
    Huffman<char> huffman_test(freqs_test);

    uint64_t encoded_data = 0;
    size_t bit_offset = 0;
    std::string test_string = "abracadabra";
    for (auto c: test_string) {
        encoded_data |= huffman_test[c].code << bit_offset;
        bit_offset += huffman_test[c].length;
    }
    bit_offset = 0;
    for (auto c: test_string)
        std::cout << huffman_test.decode(&encoded_data, bit_offset);
    std::cout << std::endl;

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
        return 1;
    }

    learnedretrieval::DatasetReader reader(argv[1]);
    learnedretrieval::DatasetReader::Row row;
    std::vector<size_t> labels_counts(reader.num_classes());
    size_t row_count = 0;
    size_t total_bits_learned_retrieval = 0;
    while (reader.next_row(row)) {
        ++row_count;
        ++labels_counts[row.label];
        Huffman<uint16_t> huff(row.probabilities);
        total_bits_learned_retrieval += huff[row.label].length;
    }
    std::cout << "Read " << row_count << " rows" << std::endl;
    std::cout << "LearnedRetrieval total bits: " << total_bits_learned_retrieval << std::endl;
    std::cout << "LearnedRetrieval bits/value: " << (double) total_bits_learned_retrieval / row_count << std::endl;

    Huffman<uint16_t> huff(labels_counts);
    size_t total_bits = 0;
    for (auto label = 0; label < labels_counts.size(); ++label)
        total_bits += huff[label].length * labels_counts[label];
    std::cout << "Huffman on values total bits: " << total_bits << std::endl;
    std::cout << "Huffman of values bits/value: " << (double) total_bits / row_count << std::endl;

    return 0;
}
