#include <atomic>
#include <cstdlib>
#include <thread>
#include <iostream>
#include <tlx/cmdline_parser.hpp>
#include <filesystem>

#include "ribbon.hpp"
#include "serialization.hpp"
#include "rocksdb/stop_watch.h"

#include "learnedretrieval/learned_retrieval.hpp"

std::string rootDir = "lrdata/";
std::string modelInput = "all";
std::string dataSetInput = "all";
std::string storageInput = "all";


void printResult(const std::vector<std::string> &benchOutput) {
    std::cout << std::endl << "RESULT ";
    for (auto s: benchOutput) {
        std::cout << s << " ";
    }
    std::cout << std::endl;
};


template<typename S, typename F>
class fano50 : public learnedretrieval::Filter50PercentWrapper<learnedretrieval::FilterFanoCoder, S, F> {
};

template<typename Storage, typename Model>
void bemchmark(const learnedretrieval::BinaryDatasetReader &dataset, Model &model, std::vector<std::string> benchOutput) {

    learnedretrieval::LearnedRetrieval<Model, Storage> lr(dataset, model);
    rocksdb::StopWatchNano timer(true);
    volatile auto sum = 0;
    timer.Start();
    for (size_t i = 0; i < dataset.size(); ++i) {
        auto example = dataset.get_example(i);
        auto label = dataset.get_label(i);
        auto output = lr.query_probabilities(example);
        sum += output[label];
    }
    auto nanos = timer.ElapsedNanos(true);
    std::cout << "Model inference time: " << nanos << " ns (" << (nanos / static_cast<double>(dataset.size()))
              << " ns/query)\n";

    timer.Start();
    for (size_t i = 0; i < dataset.size(); ++i) {
        auto example = dataset.get_example(i);
        auto label = dataset.get_label(i);
        auto output = lr.query_probabilities(example);
        sum += output[label] + lr.query_storage(i, example).first;
    }
    nanos = timer.ElapsedNanos(true);
    std::cout << "Model inference and retrieval time " << nanos << " ns ("
              << (nanos / static_cast<double>(dataset.size())) << " ns/query)\n";

    timer.Start();

    bool ok = true;
    for (size_t i = 0; i < dataset.size(); ++i) {
        auto example = dataset.get_example(i);
        auto label = dataset.get_label(i);
        uint64_t res = lr.query(i, example);
        bool found = res == label;
        assert(found);
        ok &= found;
    }
    if (!ok)
        std::cerr << "FAILED\n";
    nanos = timer.ElapsedNanos(true);
    std::cout << "Total query time: " << nanos << " ns (" << (nanos / static_cast<double>(dataset.size()))
              << " ns/query)\n";
    std::cout << learnedretrieval::myfilterbits << " " << learnedretrieval::maxFilterCnt << std::endl;

    printResult(benchOutput);
}

template<typename Model>
void dispatchStorage(const learnedretrieval::BinaryDatasetReader &dataset, Model &model, std::vector<std::string> benchOutput) {
    bool allStorage = storageInput == "all";

    if (allStorage or storageInput == "filter_huf") {
        bemchmark<learnedretrieval::FilteredRetrievalStorage<learnedretrieval::FilterCoding<learnedretrieval::FilterHuffmanCoder>>, Model>(
                dataset, model, benchOutput);
    }
    if (allStorage or storageInput == "filter_fano") {
        bemchmark<learnedretrieval::FilteredRetrievalStorage<learnedretrieval::FilterCoding<learnedretrieval::FilterFanoCoder>>, Model>(
                dataset, model, benchOutput);
    }
    if (allStorage or storageInput == "filter_fano50") {
        bemchmark<learnedretrieval::FilteredRetrievalStorage<learnedretrieval::FilterCoding<fano50>>, Model>(dataset,
                                                                                                             model, benchOutput);
    }
}


void dispatchModel(const std::string &datasetName) {
    typedef learnedretrieval::ModelWrapper<float> Model;

    learnedretrieval::BinaryDatasetReader dataset(rootDir + datasetName);

    std::vector<std::string> benchOutput;

    std::cout << "Dataset:" << std::endl
              << "  Examples: " << dataset.size() << std::endl
              << "  Features: " << dataset.features_count() << std::endl
              << "  Classes:  " << dataset.classes_count() << std::endl;

    if (modelInput == "all") {
        for (const auto &entry: std::filesystem::directory_iterator(rootDir)) {
            const std::filesystem::path &p = entry.path();
            std::string fileName = p.filename().string();
            if (fileName.starts_with(datasetName) and fileName.ends_with(".tflite")) {
                Model model(p);
                dispatchStorage<Model>(dataset, model, benchOutput);
            }
        }
    } else {
        Model model(rootDir + modelInput);
        dispatchStorage<Model>(dataset, model, benchOutput);
    }
}


void dispatchDataSet() {
    if (dataSetInput == "all") {
        for (const auto &entry: std::filesystem::directory_iterator(rootDir)) {
            std::string p = entry.path().filename().string();
            if (p.ends_with("y.lrbin")) {
                dispatchModel(p.substr(0, p.find('_')));
            }
        }
    } else {
        dispatchModel(dataSetInput);
    }
}

int main(int argc, char *argv[]) {
    tlx::CmdlineParser cmd;
    cmd.add_string('r', "rootDir", rootDir, "Path to the directory containing mdata and models");
    cmd.add_string('d', "datasetPath", dataSetInput, "Name of dataset or all");
    cmd.add_string('m', "modelPath", modelInput, "Name of model or all");
    cmd.add_string('s', "storage", storageInput, "Name of dataset or all");

    if (!cmd.process(argc, argv)) {
        cmd.print_usage();
        return EXIT_FAILURE;
    }
    dispatchDataSet();
    return EXIT_SUCCESS;
}
