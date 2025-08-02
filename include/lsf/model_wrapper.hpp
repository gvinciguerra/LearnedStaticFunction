#pragma once

#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/core/kernels/builtin_op_kernels.h"

namespace lsf {

class ModelWrapper {
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    size_t input_dims;
    size_t output_dims;
    size_t bytes;
    std::span<float> input_span;
    std::span<float> output_span;

public:

    ModelWrapper() = default;

    ModelWrapper(const std::string &model_path) {
        using namespace tflite;
        using namespace tflite::ops::builtin;
        model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
        tflite::MutableOpResolver resolver;
        resolver.AddBuiltin(BuiltinOperator_ABS, Register_ABS(), 1, 5);
        resolver.AddBuiltin(BuiltinOperator_HARD_SWISH, Register_HARD_SWISH());
        resolver.AddBuiltin(BuiltinOperator_RELU, Register_RELU(), 1, 3);
        resolver.AddBuiltin(BuiltinOperator_RELU_N1_TO_1, Register_RELU_N1_TO_1());
        resolver.AddBuiltin(BuiltinOperator_RELU_0_TO_1, Register_RELU_0_TO_1());
        resolver.AddBuiltin(BuiltinOperator_RELU6, Register_RELU6(), 1, 3);
        resolver.AddBuiltin(BuiltinOperator_TANH, Register_TANH(), 1, 3);
        resolver.AddBuiltin(BuiltinOperator_LOGISTIC, Register_LOGISTIC(), 1, 3);
        resolver.AddBuiltin(BuiltinOperator_AVERAGE_POOL_2D, Register_AVERAGE_POOL_2D(), 1, 3);
        resolver.AddBuiltin(BuiltinOperator_MAX_POOL_2D, Register_MAX_POOL_2D(), 1, 3);
        resolver.AddBuiltin(BuiltinOperator_L2_POOL_2D, Register_L2_POOL_2D());
        // resolver.AddBuiltin(BuiltinOperator_CONV_2D, Register_CONV_2D(), 1, 8);
        // resolver.AddBuiltin(BuiltinOperator_DEPTHWISE_CONV_2D, Register_DEPTHWISE_CONV_2D(), 1, 7);
        resolver.AddBuiltin(BuiltinOperator_SVDF, Register_SVDF(), 1, 4);
        resolver.AddBuiltin(BuiltinOperator_RNN, Register_RNN(), 1, 3);
        resolver.AddBuiltin(BuiltinOperator_BIDIRECTIONAL_SEQUENCE_RNN, Register_BIDIRECTIONAL_SEQUENCE_RNN(), 1, 3);
        resolver.AddBuiltin(BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN, Register_UNIDIRECTIONAL_SEQUENCE_RNN(), 1, 3);
        resolver.AddBuiltin(BuiltinOperator_EMBEDDING_LOOKUP, Register_EMBEDDING_LOOKUP(), 1, 4);
        resolver.AddBuiltin(BuiltinOperator_EMBEDDING_LOOKUP_SPARSE, Register_EMBEDDING_LOOKUP_SPARSE());
        resolver.AddBuiltin(BuiltinOperator_FULLY_CONNECTED, Register_FULLY_CONNECTED(), 1, 13);
        resolver.AddBuiltin(BuiltinOperator_LSH_PROJECTION, Register_LSH_PROJECTION());
        resolver.AddBuiltin(BuiltinOperator_HASHTABLE_LOOKUP, Register_HASHTABLE_LOOKUP());
        resolver.AddBuiltin(BuiltinOperator_SOFTMAX, Register_SOFTMAX(), 1, 3);
        resolver.AddBuiltin(BuiltinOperator_CONCATENATION, Register_CONCATENATION(), 1, 4);
        resolver.AddBuiltin(BuiltinOperator_ADD, Register_ADD(), 1, 5);
        resolver.AddBuiltin(BuiltinOperator_SPACE_TO_BATCH_ND, Register_SPACE_TO_BATCH_ND(), 1, 4);
        resolver.AddBuiltin(BuiltinOperator_BATCH_TO_SPACE_ND, Register_BATCH_TO_SPACE_ND(), 1, 4);
        resolver.AddBuiltin(BuiltinOperator_MUL, Register_MUL(), 1, 7);
        resolver.AddBuiltin(BuiltinOperator_L2_NORMALIZATION, Register_L2_NORMALIZATION(), 1, 2);
        resolver.AddBuiltin(BuiltinOperator_LOCAL_RESPONSE_NORMALIZATION, Register_LOCAL_RESPONSE_NORMALIZATION());
        resolver.AddBuiltin(BuiltinOperator_LSTM, Register_LSTM(), 1, 4);
        resolver.AddBuiltin(BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM, Register_BIDIRECTIONAL_SEQUENCE_LSTM(), 1, 3);
        resolver.AddBuiltin(BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM, Register_UNIDIRECTIONAL_SEQUENCE_LSTM(), 1, 4);
        resolver.AddBuiltin(BuiltinOperator_PAD, Register_PAD(), 1, 4);
        resolver.AddBuiltin(BuiltinOperator_PADV2, Register_PADV2(), 1, 4);
        resolver.AddBuiltin(BuiltinOperator_RESHAPE, Register_RESHAPE());
        resolver.AddBuiltin(BuiltinOperator_RESIZE_BILINEAR, Register_RESIZE_BILINEAR(), 1, 4);
        resolver.AddBuiltin(BuiltinOperator_RESIZE_NEAREST_NEIGHBOR, Register_RESIZE_NEAREST_NEIGHBOR(), 1, 4);
        resolver.AddBuiltin(BuiltinOperator_SKIP_GRAM, Register_SKIP_GRAM());
        resolver.AddBuiltin(BuiltinOperator_SPACE_TO_DEPTH, Register_SPACE_TO_DEPTH(), 1, 2);
        resolver.AddBuiltin(BuiltinOperator_DEPTH_TO_SPACE, Register_DEPTH_TO_SPACE(), 1, 2);
        resolver.AddBuiltin(BuiltinOperator_GATHER, Register_GATHER(), 1, 7);
        resolver.AddBuiltin(BuiltinOperator_TRANSPOSE, Register_TRANSPOSE(), 1, 6);
        resolver.AddBuiltin(BuiltinOperator_MEAN, Register_MEAN(), 1, 3);
        resolver.AddBuiltin(BuiltinOperator_DIV, Register_DIV(), 1, 2);
        resolver.AddBuiltin(BuiltinOperator_SUB, Register_SUB(), 1, 5);
        resolver.AddBuiltin(BuiltinOperator_SPLIT, Register_SPLIT(), 1, 4);
        resolver.AddBuiltin(BuiltinOperator_SPLIT_V, Register_SPLIT_V(), 1, 2);
        resolver.AddBuiltin(BuiltinOperator_SQUEEZE, Register_SQUEEZE(), 1, 2);
        resolver.AddBuiltin(BuiltinOperator_STRIDED_SLICE, Register_STRIDED_SLICE(), 1, 8);
        resolver.AddBuiltin(BuiltinOperator_EXP, Register_EXP(), 1, 2);
        resolver.AddBuiltin(BuiltinOperator_TOPK_V2, Register_TOPK_V2(), 1, 3);
        resolver.AddBuiltin(BuiltinOperator_LOG, Register_LOG(), 1, 2);
        resolver.AddBuiltin(BuiltinOperator_LOG_SOFTMAX, Register_LOG_SOFTMAX(), 1, 2);
        resolver.AddBuiltin(BuiltinOperator_CAST, Register_CAST(), 1, 7);
        resolver.AddBuiltin(BuiltinOperator_DEQUANTIZE, Register_DEQUANTIZE(), 1, 6);
        resolver.AddBuiltin(BuiltinOperator_PRELU, Register_PRELU());
        resolver.AddBuiltin(BuiltinOperator_MAXIMUM, Register_MAXIMUM(), 1, 4);
        resolver.AddBuiltin(BuiltinOperator_MINIMUM, Register_MINIMUM(), 1, 4);
        resolver.AddBuiltin(BuiltinOperator_ARG_MAX, Register_ARG_MAX(), 1, 3);
        resolver.AddBuiltin(BuiltinOperator_ARG_MIN, Register_ARG_MIN(), 1, 3);
        resolver.AddBuiltin(BuiltinOperator_GREATER, Register_GREATER(), 1, 2);
        resolver.AddBuiltin(BuiltinOperator_GREATER_EQUAL, Register_GREATER_EQUAL(), 1, 3);
        resolver.AddBuiltin(BuiltinOperator_LESS, Register_LESS(), 1, 3);
        resolver.AddBuiltin(BuiltinOperator_LESS_EQUAL, Register_LESS_EQUAL(), 1, 2);
        resolver.AddBuiltin(BuiltinOperator_FLOOR, Register_FLOOR());
        resolver.AddBuiltin(BuiltinOperator_CEIL, Register_CEIL());
        resolver.AddBuiltin(BuiltinOperator_ROUND, Register_ROUND());
        resolver.AddBuiltin(BuiltinOperator_NEG, Register_NEG());
        resolver.AddBuiltin(BuiltinOperator_SELECT, Register_SELECT(), 1, 4);
        resolver.AddBuiltin(BuiltinOperator_SELECT_V2, Register_SELECT_V2(), 1, 2);
        resolver.AddBuiltin(BuiltinOperator_SLICE, Register_SLICE(), 1, 6);
        resolver.AddBuiltin(BuiltinOperator_SIN, Register_SIN());
        resolver.AddBuiltin(BuiltinOperator_COS, Register_COS());
        // resolver.AddBuiltin(BuiltinOperator_TRANSPOSE_CONV, Register_TRANSPOSE_CONV(), 1, 5);
        resolver.AddBuiltin(BuiltinOperator_TILE, Register_TILE(), 1, 3);
        resolver.AddBuiltin(BuiltinOperator_SUM, Register_SUM(), 1, 2);
        resolver.AddBuiltin(BuiltinOperator_REDUCE_PROD, Register_REDUCE_PROD(), 1, 2);
        resolver.AddBuiltin(BuiltinOperator_REDUCE_MAX, Register_REDUCE_MAX(), 1, 3);
        resolver.AddBuiltin(BuiltinOperator_REDUCE_MIN, Register_REDUCE_MIN(), 1, 3);
        resolver.AddBuiltin(BuiltinOperator_REDUCE_ANY, Register_REDUCE_ANY());
        resolver.AddBuiltin(BuiltinOperator_REDUCE_ALL, Register_REDUCE_ALL());
        resolver.AddBuiltin(BuiltinOperator_EXPAND_DIMS, Register_EXPAND_DIMS());
        resolver.AddBuiltin(BuiltinOperator_SPARSE_TO_DENSE, Register_SPARSE_TO_DENSE(), 1, 3);
        resolver.AddBuiltin(BuiltinOperator_EQUAL, Register_EQUAL(), 1, 4);
        resolver.AddBuiltin(BuiltinOperator_NOT_EQUAL, Register_NOT_EQUAL(), 1, 3);
        resolver.AddBuiltin(BuiltinOperator_SQRT, Register_SQRT());
        resolver.AddBuiltin(BuiltinOperator_RSQRT, Register_RSQRT(), 1, 3);
        resolver.AddBuiltin(BuiltinOperator_SHAPE, Register_SHAPE());
        resolver.AddBuiltin(BuiltinOperator_RANK, Register_RANK());
        resolver.AddBuiltin(BuiltinOperator_POW, Register_POW());
        resolver.AddBuiltin(BuiltinOperator_FAKE_QUANT, Register_FAKE_QUANT(), 1, 2);
        resolver.AddBuiltin(BuiltinOperator_PACK, Register_PACK(), 1, 4);
        resolver.AddBuiltin(BuiltinOperator_ONE_HOT, Register_ONE_HOT());
        resolver.AddBuiltin(BuiltinOperator_LOGICAL_OR, Register_LOGICAL_OR());
        resolver.AddBuiltin(BuiltinOperator_LOGICAL_AND, Register_LOGICAL_AND());
        resolver.AddBuiltin(BuiltinOperator_LOGICAL_NOT, Register_LOGICAL_NOT());
        resolver.AddBuiltin(BuiltinOperator_UNPACK, Register_UNPACK(), 1, 4);
        resolver.AddBuiltin(BuiltinOperator_FLOOR_DIV, Register_FLOOR_DIV(), 1, 3);
        resolver.AddBuiltin(BuiltinOperator_SQUARE, Register_SQUARE());
        resolver.AddBuiltin(BuiltinOperator_ZEROS_LIKE, Register_ZEROS_LIKE());
        resolver.AddBuiltin(BuiltinOperator_FLOOR_MOD, Register_FLOOR_MOD(), 1, 2);
        resolver.AddBuiltin(BuiltinOperator_RANGE, Register_RANGE(), 1, 2);
        resolver.AddBuiltin(BuiltinOperator_LEAKY_RELU, Register_LEAKY_RELU(), 1, 2);
        resolver.AddBuiltin(BuiltinOperator_SQUARED_DIFFERENCE, Register_SQUARED_DIFFERENCE(), 1, 2);
        resolver.AddBuiltin(BuiltinOperator_FILL, Register_FILL(), 1, 4);
        resolver.AddBuiltin(BuiltinOperator_MIRROR_PAD, Register_MIRROR_PAD(), 1, 3);
        resolver.AddBuiltin(BuiltinOperator_UNIQUE, Register_UNIQUE());
        resolver.AddBuiltin(BuiltinOperator_REVERSE_V2, Register_REVERSE_V2(), 1, 3);
        resolver.AddBuiltin(BuiltinOperator_ADD_N, Register_ADD_N());
        resolver.AddBuiltin(BuiltinOperator_GATHER_ND, Register_GATHER_ND(), 1, 5);
        resolver.AddBuiltin(BuiltinOperator_WHERE, Register_WHERE(), 1, 2);
        resolver.AddBuiltin(BuiltinOperator_ELU, Register_ELU());
        resolver.AddBuiltin(BuiltinOperator_REVERSE_SEQUENCE, Register_REVERSE_SEQUENCE());
        resolver.AddBuiltin(BuiltinOperator_MATRIX_DIAG, Register_MATRIX_DIAG());
        resolver.AddBuiltin(BuiltinOperator_QUANTIZE, Register_QUANTIZE(), 1, 3);
        resolver.AddBuiltin(BuiltinOperator_MATRIX_SET_DIAG, Register_MATRIX_SET_DIAG());
        resolver.AddBuiltin(BuiltinOperator_IF, tflite::ops::builtin::Register_IF());
        resolver.AddBuiltin(BuiltinOperator_WHILE, tflite::ops::builtin::Register_WHILE());
        resolver.AddBuiltin(BuiltinOperator_NON_MAX_SUPPRESSION_V4, Register_NON_MAX_SUPPRESSION_V4());
        resolver.AddBuiltin(BuiltinOperator_NON_MAX_SUPPRESSION_V5, Register_NON_MAX_SUPPRESSION_V5());
        resolver.AddBuiltin(BuiltinOperator_SCATTER_ND, Register_SCATTER_ND());
        resolver.AddBuiltin(BuiltinOperator_DENSIFY, Register_DENSIFY());
        resolver.AddBuiltin(BuiltinOperator_SEGMENT_SUM, Register_SEGMENT_SUM());
        resolver.AddBuiltin(BuiltinOperator_BATCH_MATMUL, Register_BATCH_MATMUL(), 1, 4);
        resolver.AddBuiltin(BuiltinOperator_CUMSUM, Register_CUMSUM());
        resolver.AddBuiltin(BuiltinOperator_BROADCAST_TO, Register_BROADCAST_TO(), 2, 3);
        resolver.AddBuiltin(BuiltinOperator_CALL_ONCE, tflite::ops::builtin::Register_CALL_ONCE());
        resolver.AddBuiltin(BuiltinOperator_RFFT2D, Register_RFFT2D());
        // resolver.AddBuiltin(BuiltinOperator_CONV_3D, Register_CONV_3D());
        resolver.AddBuiltin(BuiltinOperator_IMAG, Register_IMAG());
        resolver.AddBuiltin(BuiltinOperator_REAL, Register_REAL());
        resolver.AddBuiltin(BuiltinOperator_COMPLEX_ABS, Register_COMPLEX_ABS());
        resolver.AddBuiltin(BuiltinOperator_BROADCAST_ARGS, Register_BROADCAST_ARGS());
        resolver.AddBuiltin(BuiltinOperator_HASHTABLE, Register_HASHTABLE());
        resolver.AddBuiltin(BuiltinOperator_HASHTABLE_FIND, Register_HASHTABLE_FIND());
        resolver.AddBuiltin(BuiltinOperator_HASHTABLE_IMPORT, Register_HASHTABLE_IMPORT());
        resolver.AddBuiltin(BuiltinOperator_HASHTABLE_SIZE, Register_HASHTABLE_SIZE());
        // resolver.AddBuiltin(BuiltinOperator_CONV_3D_TRANSPOSE, Register_CONV_3D_TRANSPOSE());
        resolver.AddBuiltin(BuiltinOperator_VAR_HANDLE, Register_VAR_HANDLE());
        resolver.AddBuiltin(BuiltinOperator_READ_VARIABLE, Register_READ_VARIABLE());
        resolver.AddBuiltin(BuiltinOperator_ASSIGN_VARIABLE, Register_ASSIGN_VARIABLE());
        resolver.AddBuiltin(BuiltinOperator_MULTINOMIAL, Register_MULTINOMIAL());
        resolver.AddBuiltin(BuiltinOperator_RANDOM_STANDARD_NORMAL, Register_RANDOM_STANDARD_NORMAL());
        resolver.AddBuiltin(BuiltinOperator_BUCKETIZE, Register_BUCKETIZE());
        resolver.AddBuiltin(BuiltinOperator_RANDOM_UNIFORM, Register_RANDOM_UNIFORM());
        resolver.AddBuiltin(BuiltinOperator_GELU, Register_GELU(), 1, 2);
        resolver.AddBuiltin(BuiltinOperator_DYNAMIC_UPDATE_SLICE, Register_DYNAMIC_UPDATE_SLICE(), 1, 3);
        resolver.AddBuiltin(BuiltinOperator_UNSORTED_SEGMENT_PROD, Register_UNSORTED_SEGMENT_PROD());
        resolver.AddBuiltin(BuiltinOperator_UNSORTED_SEGMENT_MAX, Register_UNSORTED_SEGMENT_MAX());
        resolver.AddBuiltin(BuiltinOperator_UNSORTED_SEGMENT_MIN, Register_UNSORTED_SEGMENT_MIN());
        resolver.AddBuiltin(BuiltinOperator_UNSORTED_SEGMENT_SUM, Register_UNSORTED_SEGMENT_SUM());
        resolver.AddBuiltin(BuiltinOperator_ATAN2, Register_ATAN2());
        resolver.AddBuiltin(BuiltinOperator_SIGN, Register_SIGN(), 1, 2);
        resolver.AddBuiltin(BuiltinOperator_BITCAST, Register_BITCAST());
        resolver.AddBuiltin(BuiltinOperator_BITWISE_XOR, Register_BITWISE_XOR());
        resolver.AddBuiltin(BuiltinOperator_RIGHT_SHIFT, Register_RIGHT_SHIFT());
        resolver.AddBuiltin(BuiltinOperator_STABLEHLO_SCATTER, Register_STABLEHLO_SCATTER());
        resolver.AddBuiltin(BuiltinOperator_DILATE, Register_DILATE());
        resolver.AddBuiltin(BuiltinOperator_STABLEHLO_RNG_BIT_GENERATOR, Register_STABLEHLO_RNG_BIT_GENERATOR());
        resolver.AddBuiltin(BuiltinOperator_REDUCE_WINDOW, Register_REDUCE_WINDOW());
        resolver.AddBuiltin(BuiltinOperator_STABLEHLO_REDUCE_WINDOW, Register_STABLEHLO_REDUCE_WINDOW());
        resolver.AddBuiltin(BuiltinOperator_STABLEHLO_GATHER, Register_STABLEHLO_GATHER());
        resolver.AddBuiltin(BuiltinOperator_STABLEHLO_ADD, Register_STABLEHLO_ADD());
        resolver.AddBuiltin(BuiltinOperator_STABLEHLO_AND, Register_STABLEHLO_AND());
        resolver.AddBuiltin(BuiltinOperator_STABLEHLO_MULTIPLY, Register_STABLEHLO_MULTIPLY());
        resolver.AddBuiltin(BuiltinOperator_STABLEHLO_MAXIMUM, Register_STABLEHLO_MAXIMUM());
        resolver.AddBuiltin(BuiltinOperator_STABLEHLO_MINIMUM, Register_STABLEHLO_MINIMUM());
        resolver.AddBuiltin(BuiltinOperator_STABLEHLO_SHIFT_LEFT, Register_STABLEHLO_SHIFT_LEFT());
        resolver.AddBuiltin(BuiltinOperator_STABLEHLO_PAD, Register_STABLEHLO_PAD());
        resolver.AddBuiltin(BuiltinOperator_STABLEHLO_COMPOSITE, Register_STABLEHLO_COMPOSITE());
        resolver.AddBuiltin(BuiltinOperator_STABLEHLO_CASE, Register_STABLEHLO_CASE());
        tflite::InterpreterBuilder builder(*model, resolver);
        builder(&interpreter);
        interpreter->SetNumThreads(1);
        interpreter->AllocateTensors();
        //    printf("=== Pre-invoke Interpreter State ===\n");
        //    tflite::PrintInterpreterState(interpreter.get());

        input_dims = interpreter->input_tensor(0)->dims[1].data[0];
        output_dims = interpreter->output_tensor(0)->dims[1].data[0];

        if (interpreter->output_tensor(0)->type != kTfLiteFloat32) {
            throw std::runtime_error("Unsupported input type");
        }
        if (interpreter->output_tensor(0)->type != kTfLiteFloat32) {
            throw std::runtime_error("Unsupported input type");
        }

        bytes = 0;
        for (int i = 0; i < interpreter->tensors_size(); i++) {
            if (interpreter->tensor(i)->allocation_type == kTfLiteMmapRo) {
                bytes += interpreter->tensor(i)->bytes;
            }
        }

        input_span = std::span(interpreter->typed_input_tensor<float>(0), input_dims);
        output_span = std::span(interpreter->typed_output_tensor<float>(0), std::max<size_t>(2, output_dims));
    }

    size_t model_bytes() const { return bytes; }

    std::span<float> invoke(std::span<const float> example) {
        std::copy(example.begin(), example.end(), input_span.begin());
        interpreter->Invoke();
        if (output_dims == 1) {
            output_span[1] = output_span[0];
            output_span[0] = 1 - output_span[0];
        }
        return output_span;
    }

};
}
