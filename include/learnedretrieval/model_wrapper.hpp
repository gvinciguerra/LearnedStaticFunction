#pragma once

#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"

namespace learnedretrieval {
template <typename T>
class ModelWrapper {
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    size_t input_dims;
    size_t output_dims;
    size_t bytes;
    std::span<float> input_span;
    std::span<T> output_span;
    std::vector<float> rescaled_output;
    TfLiteQuantizationParams quantization_params;

    static_assert(std::is_same_v<T, float> || std::is_same_v<T, uint8_t>,
                  "ModelWrapper only supports float and uint8_t types.");

public:

    ModelWrapper(const std::string &model_path) {
        if constexpr (std::is_same_v<T, float>) {
            model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
        } else if constexpr (std::is_same_v<T, uint8_t>) {
            std::string uint8_model_path = model_path;
            auto pos = uint8_model_path.rfind(".tflite");
            if (pos != std::string::npos) {
                uint8_model_path.replace(pos, 7, "_uint8.tflite");
            } else {
                throw std::runtime_error("Invalid model path: " + model_path);
            }
            model = tflite::FlatBufferModel::BuildFromFile(uint8_model_path.c_str());
        }
        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder builder(*model, resolver);
        builder(&interpreter);
        interpreter->SetNumThreads(1);
        interpreter->AllocateTensors();
        //    printf("=== Pre-invoke Interpreter State ===\n");
        //    tflite::PrintInterpreterState(interpreter.get());

        input_dims = interpreter->input_tensor(0)->dims[1].data[0];
        output_dims = interpreter->output_tensor(0)->dims[1].data[0];

        if constexpr (std::is_same_v<T, float>) {
            if (interpreter->output_tensor(0)->type != kTfLiteFloat32) {
                throw std::runtime_error("Unsupported input type");
            }
        } else if constexpr (std::is_same_v<T, uint8_t>) {
            if (interpreter->output_tensor(0)->type != kTfLiteUInt8) {
                throw std::runtime_error("Unsupported input type");
            }
            quantization_params = interpreter->output_tensor(0)->params;
            rescaled_output.resize(output_dims);
        }


        bytes = 0;
        for (int i = 0; i < interpreter->tensors_size(); i++) {
            if (interpreter->tensor(i)->allocation_type == kTfLiteMmapRo) {
                bytes += interpreter->tensor(i)->bytes;
            }
        }

        input_span = std::span<float>(interpreter->typed_input_tensor<float>(0), input_dims);
        output_span = std::span<T>(interpreter->typed_output_tensor<T>(0), output_dims);
    }

    size_t model_bytes() const { return bytes; }

    std::span<T> invoke(std::span<const float> example) {
        std::copy(example.begin(), example.end(), input_span.begin());
        interpreter->Invoke();
        return output_span;
    }

    std::span<T> get_output() {
        return output_span;
    }

    std::span<float> get_probabilities() {
        if constexpr (std::is_same_v<T, uint8_t>) {
            for (size_t i = 0; i < output_dims; ++i)
                rescaled_output[i] = (output_span[i] - quantization_params.zero_point) * quantization_params.scale;
            return std::span<float>(rescaled_output.data(), output_dims);
        } else {
            return output_span;
        }
    }

};
}