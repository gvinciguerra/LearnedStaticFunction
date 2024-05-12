#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

/** Class implementing Huffman coding, a prefix code that is optimal among codes that assign an integral number of bits
 * to each symbol. */
template<typename Symbol>
class Huffman {
    struct Node {
        Symbol symbol;
        float frequency;
        Node *left;
        Node *right;
        Node *next;
    };
    struct Code {
        uint64_t code;  ///< The value of the code.
        uint8_t length; ///< The length of the code in bits.
    };
    Node *root;
    std::unordered_map<Symbol, Code> codes;

public:

    Huffman() = default;

    /** Constructs a Huffman tree from a vector of symbols and their frequencies. */
    template<typename Frequency>
    explicit Huffman(const std::vector<std::pair<Symbol, Frequency>> &input) { initialize(input); }

    /** Constructs a Huffman tree from a vector of frequencies. The symbols are the indices of the vector. */
    template<typename FrequenciesVector>
    explicit Huffman(const FrequenciesVector &input) {
        std::vector<std::pair<Symbol, typename FrequenciesVector::value_type>> freqs(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            freqs[i].first = i;
            freqs[i].second = input[i];
        }
        initialize(freqs);
    }

    /** Decode a symbol from a bitstream. */
    Symbol decode(const uint64_t *data, size_t &bit_offset) {
        auto node = root;
        while (node->left != nullptr) {
            if (data[bit_offset / 64] & (uint64_t(1) << (bit_offset % 64)))
                node = node->left;
            else
                node = node->right;
            ++bit_offset;
        }
        return node->symbol;
    }

    /** Returns the code for a symbol. */
    Code operator[](Symbol symbol) const { return codes.at(symbol); }

    ~Huffman() { delete_tree(root); }

    Huffman(const Huffman &) = delete;

    Huffman &operator=(const Huffman &) = delete;

    Huffman(Huffman &&) = delete;

    Huffman &operator=(Huffman &&) = delete;

private:

    template<typename Frequency>
    void initialize(std::vector<std::pair<Symbol, Frequency>> input) {
        // Algorithm 2.2 from the "Compact Data Structures" book by Gonzalo Navarro
        std::sort(input.begin(), input.end(), [](const auto &a, const auto &b) { return a.second < b.second; });

        root = new Node{input[0].first, (float) input[0].second, nullptr, nullptr, nullptr};
        auto prev = root;
        for (size_t i = 1; i < input.size(); ++i) {
            auto new_node = new Node{input[i].first, (float) input[i].second, nullptr, nullptr, nullptr};
            prev->next = new_node;
            prev = new_node;
        }

        auto i = root;
        while (root->next) {
            auto node = new Node{0, root->frequency + root->next->frequency, root, root->next, nullptr};
            while (i->next && i->next->frequency < node->frequency)
                i = i->next;
            node->next = i->next;
            i->next = node;
            root = root->next->next;
        }

        build_codes(root, 0, 0);
    }

    void build_codes(Node *node, uint64_t code, uint8_t length) {
        if (node->left == nullptr && node->right == nullptr) {
            codes[node->symbol] = {code, length};
            return;
        }
        if (length + 1 > 64)
            throw std::runtime_error("Code length exceeds 64 bits.");
        build_codes(node->left, code | (uint64_t(1) << length), length + 1);
        build_codes(node->right, code, length + 1);
    }

    void delete_tree(Node *node) {
        if (node == nullptr)
            return;
        delete_tree(node->left);
        delete_tree(node->right);
        delete node;
    }
};