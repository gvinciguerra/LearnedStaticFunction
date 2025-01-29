#pragma once

#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"

namespace learnedretrieval {
class ModelWrapper {
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    size_t input_dims;
    size_t output_dims;
    size_t bytes;
    std::span<float> output_span;

public:

    ModelWrapper(const std::string &model_path) {
        model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder builder(*model, resolver);
        builder(&interpreter);
        interpreter->SetNumThreads(1);
        interpreter->AllocateTensors();
        //    printf("=== Pre-invoke Interpreter State ===\n");
        //    tflite::PrintInterpreterState(interpreter.get());

        input_dims = interpreter->input_tensor(0)->dims[1].data[0];
        output_dims = interpreter->output_tensor(0)->dims[1].data[0];

        bytes = 0;
        for (int i = 0; i < interpreter->tensors_size(); i++) {
            if (interpreter->tensor(i)->allocation_type == kTfLiteMmapRo) {
                bytes += interpreter->tensor(i)->bytes;
            }
        }

        output_span = std::span<float>(interpreter->typed_output_tensor<float>(0), output_dims);
    }

    size_t model_bytes() const { return bytes; }

    std::span<float> invoke(std::span<const float> example) {
        auto input = interpreter->typed_input_tensor<float>(0);
        std::copy(example.begin(), example.end(), input);
        interpreter->Invoke();
        return output_span;
    }

};
}