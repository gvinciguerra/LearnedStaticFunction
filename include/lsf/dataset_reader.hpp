#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <span>
#include <vector>

namespace lsf {
class BinaryDatasetReader {
    using label_type = uint16_t;
    std::vector<label_type> labels;
    std::vector<float> examples;
    size_t num_examples;
    size_t num_features;
    size_t num_classes;

public:
    BinaryDatasetReader() = default;

    BinaryDatasetReader(const std::string &path) {
        auto examples_path = path + "_X.lrbin";
        auto labels_path = path + "_y.lrbin";

        std::ifstream file(examples_path, std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error("Could not open examples file at " + examples_path);

        file.read(reinterpret_cast<char *>(&num_examples), sizeof(size_t));
        file.read(reinterpret_cast<char *>(&num_features), sizeof(size_t));
        examples.resize(num_examples * num_features);
        file.read(reinterpret_cast<char *>(&examples[0]), num_examples * num_features * sizeof(float));
        file.close();

        file.open(labels_path, std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error("Could not open labels file at " + labels_path);

        labels.resize(num_examples);
        label_type n_classes;
        file.read(reinterpret_cast<char *>(&n_classes), sizeof(label_type));
        file.read(reinterpret_cast<char *>(&labels[0]), num_examples * sizeof(label_type));
        num_classes = n_classes;
    }

    size_t size() const { return num_examples; }

    size_t features_count() const { return num_features; }

    size_t classes_count() const { return num_classes; }

    std::span<const float> get_example(size_t i) const { return {&examples[i * num_features], num_features}; }

    label_type get_label(size_t i) const { return labels[i]; }

    const std::vector<label_type> &get_labels() const { return labels; }
};
}
