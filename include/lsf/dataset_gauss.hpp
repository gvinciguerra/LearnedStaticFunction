#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <span>
#include <vector>
#include <random>

namespace lsf {
class GaussDataset {
    using label_type = uint16_t;
    std::vector<label_type> labels;
    std::vector<float> examples;
    size_t num_examples;
    size_t num_features;
    size_t num_classes;

public:
    GaussDataset() = default;

    GaussDataset(size_t classes, float sigma, size_t n) {
        double distance = 2.0;
        std::mt19937 rng(42);

        examples.reserve(n);
        labels.reserve(n);

        for (int i = 0; i < classes; i++) {
            double mean = i * distance;
            std::normal_distribution<double> dist(mean, sigma);

            for (long long j = 0; j < n / classes; j++) {
                examples.push_back(dist(rng));
                labels.push_back(i);
            }
        }
        num_classes = classes;
        num_features = 1;
        num_examples = n;
    }

    size_t size() const { return num_examples; }

    size_t features_count() const { return num_features; }

    size_t classes_count() const { return num_classes; }

    std::span<const float> get_example(size_t i) const { return {&examples[i * num_features], num_features}; }

    label_type get_label(size_t i) const { return labels[i]; }

    const std::vector<label_type> &get_labels() const { return labels; }
};
}
