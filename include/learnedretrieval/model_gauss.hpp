#pragma once

#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"

#include <numeric>
#include <cmath>

namespace lsf {

    class ModelGauss {
        static constexpr float start = -1.5;
        static constexpr float width = 3;

        float invtwosigmasquared;
        float step;

        std::vector<float> output;
    public:

        ModelGauss() = default;

        ModelGauss(double sigma, size_t labels) : invtwosigmasquared(-0.5f / (sigma * sigma)), step(width / (labels-1)) {
            output.resize(labels);
        }

        size_t model_bytes() const { return sizeof(invtwosigmasquared) + sizeof(output); }

        std::span<float> invoke(std::span<const float> example) {
            float pos = start;
            for (auto &v: output) {
                float delta = example[0] - pos;
                v = expf(delta * delta * invtwosigmasquared);
                pos += step;
            }

            // normalize
            float sum = std::accumulate(output.begin(), output.end(), 0.0f);
            sum = 1.0f / sum;
            for (auto &v: output) {
                v *= sum;
            }
            return output;
        }

    };
}