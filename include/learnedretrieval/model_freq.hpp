#pragma once

#include <cmath>

namespace lsf {

    // for CSF like construction
    class ModelFreq {
        std::vector<float> output;
    public:

        ModelFreq(const std::vector<uint16_t> &trainY, size_t classes_count) {
            output.resize(classes_count);
            float inc = 1.0f / float(classes_count);
            for (auto i : trainY) {
                output[i] += inc;
            }
        }

        size_t model_bytes() const { return sizeof(float) * output.size(); }

        size_t model_params_count() const { output.size(); }


        std::span<float> invoke(std::span<const float> example) {
            return output;
        }
    };
}