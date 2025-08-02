#include <atomic>
#include <cstdlib>
#include <cmath>
#include <thread>
#include <iostream>
#include <tlx/cmdline_parser.hpp>
#include <filesystem>

#include "ribbon.hpp"
#include "serialization.hpp"
#include "rocksdb/stop_watch.h"

#include "lsf/learned_static_function.hpp"
#include "lsf/model_gauss.hpp"

std::string rootDir = "lrdata/";
constexpr std::string ALL = "all";
std::string modelInput = ALL;
std::string dataSetInput = ALL;


void printResult(const std::vector<std::string> &benchOutput) {
    std::cout << std::endl << "RESULT ";
    for (auto s: benchOutput) {
        std::cout << s << " ";
    }
};

template<typename Model>
void determineCalibration(const lsf::BinaryDatasetReader &dataset, Model &model,
                          std::vector<std::string> benchOutput, std::string modelName) {
    std::cout << "### Next model: " << modelName << std::endl;


    // model
    benchOutput.push_back("model_bits=" + std::to_string(8.0 * model.model_bytes() / double(dataset.size())));
    benchOutput.push_back("model_name=" + modelName);
    double entropy = 0;
    const float min_prob = std::pow(2.f, -31.f);
    for (int i = 0; i < dataset.size(); ++i) {
        entropy -= std::log2(std::max(model.invoke(dataset.get_example(i))[dataset.get_label(i)], min_prob));
    }
    benchOutput.push_back("cross_entropy_bit_per_key=" + std::to_string(entropy / dataset.size()));

    std::vector<std::pair<double, double>> data;
    for (int i = 0; i < dataset.size(); ++i) {
        auto example = dataset.get_example(i);
        auto output = model.invoke(example);
        for (int j = 0; j < dataset.classes_count(); ++j) {
            data.push_back({output[j], j == dataset.get_label(i)});
        }
    }
    std::sort(data.begin(), data.end(), [](const auto &a, const auto &b) {
        return a.first < b.first;
    });

    size_t bucketSize = data.size() / 50;
    double x_sum = 0.0, y_sum = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        double x = data[i].first;
        double y = data[i].second;

        x_sum += x;
        y_sum += y;

        if (i % bucketSize == 0) {
            double avg_x = x_sum / double(bucketSize);
            double avg_y = y_sum / double(bucketSize);
            auto resultLine = benchOutput;
            resultLine.push_back("x=" + std::to_string(avg_x));
            resultLine.push_back("y=" + std::to_string(avg_y));
            printResult(resultLine);
            x_sum = 0.0;
            y_sum = 0.0;
        }
    }
}

void dispatchAllModelsRecurse(const std::string &datasetName, const lsf::BinaryDatasetReader &dataset,
                              const std::vector<std::string> &benchOutput, const std::string &dir) {
    for (const auto &entry: std::filesystem::directory_iterator(dir)) {
        if (entry.is_directory()) {
            dispatchAllModelsRecurse(datasetName, dataset, benchOutput,
                                     dir + "/" + entry.path().filename().string());
        } else {
            const std::filesystem::path &p = entry.path();
            std::string fileName = p.filename().string();
            if (fileName.starts_with(datasetName) and fileName.ends_with(".tflite") and
                (modelInput == ALL or fileName.contains(modelInput))) {
                try {
                    lsf::ModelWrapper model(p);
                    auto evalFile = p.string() + "_eval.txt";
                    std::ifstream evalStream;
                    evalStream.exceptions(std::ifstream::failbit | std::ifstream::badbit);
                    evalStream.open(evalFile);
                    std::string line;
                    std::getline(evalStream, line);
                    std::istringstream iss(line);
                    std::string token;
                    std::vector benchOutputCopy = benchOutput;
                    while (iss >> token)
                        benchOutputCopy.push_back(token);
                    determineCalibration<lsf::ModelWrapper>(dataset, model, benchOutputCopy, fileName);
                } catch (std::runtime_error &e) {
                    std::cerr << "Skipping model " << fileName << " because of " << e.what() << std::endl;
                }
            }
        }
    }
}

void dispatchModel(const std::string &datasetName, std::vector<std::string> benchOutput) {

    // dataset
    std::cout << "### Next dataset: " << datasetName << std::endl;
    benchOutput.push_back("dataset_name=" + datasetName);
    lsf::BinaryDatasetReader dataset(rootDir + datasetName);

    std::vector<size_t> cnt;
    cnt.resize(dataset.classes_count());
    for (int i = 0; i < dataset.size(); ++i) {
        cnt[dataset.get_label(i)]++;
    }
    auto sum = double(std::accumulate(cnt.begin(), cnt.end(), 0));
    double entropy = 0;
    for (size_t c: cnt) {
        double p = double(c) / sum;
        entropy -= p * std::log2(p);
    }
    benchOutput.push_back("entropy=" + std::to_string(entropy));
    benchOutput.push_back("size=" + std::to_string(dataset.size()));
    benchOutput.push_back("features=" + std::to_string(dataset.features_count()));
    benchOutput.push_back("classes=" + std::to_string(dataset.classes_count()));

    // model
    if (datasetName.starts_with("gauss")) {
        std::vector<uint32_t> indexes(dataset.size());
        std::iota(indexes.begin(), indexes.end(), 0);
        std::shuffle(indexes.begin(), indexes.end(), std::mt19937(42));
        auto testSize = dataset.size() / 5;
        std::vector<float> trainX, testX;
        std::vector<uint16_t> trainY, testY;
        trainX.reserve(dataset.size() - testSize);
        trainY.reserve(dataset.size() - testSize);
        testX.reserve(testSize);
        testY.reserve(testSize);
        for (size_t i = 0; i < dataset.size(); ++i) {
            auto example = dataset.get_example(indexes[i])[0];
            auto label = dataset.get_label(indexes[i]);
            if (i < testSize) {
                testX.push_back(example);
                testY.push_back(label);
            } else {
                trainX.push_back(example);
                trainY.push_back(label);
            }
        }
        rocksdb::StopWatchNano timer(true);
        lsf::ModelGaussianNaiveBayes model(trainX, trainY, dataset.classes_count());
        auto nanos = timer.ElapsedNanos(true);
        benchOutput.push_back("training_seconds=" + std::to_string(double(nanos) / 1e9));
        benchOutput.push_back("model_params=" + std::to_string(model.model_params_count()));
        benchOutput.push_back("test_accuracy=" + std::to_string(100.0f * model.eval_accuracy(testX, testY)));
        determineCalibration<lsf::ModelGaussianNaiveBayes>(dataset, model, benchOutput, "gauss");
    } else {
        dispatchAllModelsRecurse(datasetName, dataset, benchOutput, rootDir);
    }
}


void dispatchDataSet(std::vector<std::string> benchOutput) {
    if (dataSetInput == ALL) {
        for (const auto &entry: std::filesystem::directory_iterator(rootDir)) {
            std::string p = entry.path().filename().string();
            if (p.ends_with("y.lrbin")) {
                dispatchModel(p.substr(0, p.find('_')), benchOutput);
            }
        }
    } else {
        dispatchModel(dataSetInput, benchOutput);
    }
}

int main(int argc, char *argv[]) {
    tlx::CmdlineParser cmd;
    cmd.add_string('r', "rootDir", rootDir, "Path to the directory containing mdata and models");
    cmd.add_string('d', "datasetPath", dataSetInput, "Name of dataset or all");
    cmd.add_string('m', "model", modelInput,
                   "Includes all models that have the substring in their filename or all");

    if (!cmd.process(argc, argv)) {
        cmd.print_usage();
        return EXIT_FAILURE;
    }

    std::vector<std::string> benchOutput;
    dispatchDataSet(benchOutput);
    return EXIT_SUCCESS;
}
