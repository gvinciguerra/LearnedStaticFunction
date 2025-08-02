#pragma once

#include <cmath>

namespace lsf {

    class ModelGaussianNaiveBayes {
        static constexpr float sqrt_2pi = 2.5066282746f;
        struct Parameters {
            float mean;
            float std;
        };
        std::vector<Parameters> parameters;

        std::vector<float> output;
    public:

        ModelGaussianNaiveBayes(const std::vector<float> &trainX,
                                const std::vector<uint16_t> &trainY,
                                size_t classes_count) {
            std::vector<RunningStats> stats(classes_count);
            for (size_t i = 0; i < trainX.size(); ++i)
                stats[trainY[i]].push(trainX[i]);
            output.resize(classes_count);
            parameters.resize(classes_count);
            for (size_t i = 0; i < classes_count; ++i) {
                parameters[i].mean = stats[i].mean();
                parameters[i].std = stats[i].standard_deviation();
            }
        }

        size_t model_bytes() const { return sizeof(Parameters) * parameters.size(); }

        size_t model_params_count() const { return 2 * parameters.size(); }

        float eval_accuracy(const std::vector<float> &testX, const std::vector<uint16_t> &testY) {
            size_t correct = 0;
            for (size_t i = 0; i < testX.size(); ++i) {
                auto output = invoke(std::span<const float>(&testX[i], 1));
                auto max_it = std::max_element(output.begin(), output.end());
                if (std::distance(output.begin(), max_it) == testY[i])
                    correct++;
            }
            return static_cast<float>(correct) / testX.size();
        }

        std::span<float> invoke(std::span<const float> example) {
            float sum = 0.0f;
            for (size_t i = 0; i < parameters.size(); ++i) {
                auto diff = example[0] - parameters[i].mean;
                auto exponent = -0.5f * (diff * diff) / (parameters[i].std * parameters[i].std);
                output[i] = std::exp(exponent) / (parameters[i].std * sqrt_2pi);
                sum += output[i];
            }
            for (auto &o : output)
                o /= sum;
            return output;
        }

        class RunningStats {
            size_t n;
            double m_oldM;
            double m_newM;
            double m_oldS;
            double m_newS;
            double m_total;
            double m_max;
            double m_min;

        public:
            RunningStats() : n(0) {}

            void push(double x) {
                n++;
                if (n == 1) {
                    m_oldM = m_newM = x;
                    m_oldS = 0.0;
                    m_total = x;
                    m_max = x;
                    m_min = x;
                } else {
                    m_newM = m_oldM + (x - m_oldM) / n;
                    m_newS = m_oldS + (x - m_oldM) * (x - m_newM);
                    m_oldM = m_newM;
                    m_oldS = m_newS;
                    m_total += x;
                }
            }

            size_t samples() const {  return n;  }
            double mean() const { return (n > 0) ? m_newM : 0.0; }
            double variance() const { return ((n > 1) ? m_newS / (n - 1) : 0.0); }
            double standard_deviation() const { return std::sqrt(variance()); }
            double total() const { return m_total;  }
        };

    };
}