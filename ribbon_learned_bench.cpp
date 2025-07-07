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

#include "learnedretrieval/learned_retrieval.hpp"

#define QUERIES 10000000
#define REPEATS 3
#define TOT_QUERIES (QUERIES*REPEATS)

std::string rootDir = "lrdata/";
constexpr std::string ALL = "all";
std::string modelInput = ALL;
std::string dataSetInput = ALL;
std::string storageInput = ALL;

typedef learnedretrieval::ModelWrapper Model;

void printResult(const std::vector<std::string> &benchOutput) {
    std::cout << std::endl << "RESULT ";
    for (auto s: benchOutput) {
        std::cout << s << " ";
    }
    std::cout << std::endl;
};


template<typename S, typename F>
class FilteredFano50 : public learnedretrieval::Filter50PercentWrapper<learnedretrieval::FilterFanoCoder, S, F> {
};

template<typename Storage, typename Model>
void
bemchmark(const learnedretrieval::BinaryDatasetReader &dataset, Model &model, std::vector<std::string> benchOutput) {

    std::cout << "### Next storage: " << Storage::get_name() << std::endl;

    benchOutput.push_back("storage_name=" + Storage::get_name());
    rocksdb::StopWatchNano timer(true);

    learnedretrieval::LearnedRetrieval<Model, Storage> lr(dataset, model);

    auto nanos = timer.ElapsedNanos(true);
    std::cout << "Total Construct " << nanos << " ns ("
              << (nanos / static_cast<double>(dataset.size())) << " ns/key)\n";

    benchOutput.push_back("construct_ms=" + std::to_string(double(nanos) / 1000.0 / 1000.0));
    benchOutput.push_back("storage_bits=" + std::to_string(8.0 * lr.size_in_bytes() / double(dataset.size())));
    volatile uint64_t sum = 0;

    std::vector<uint32_t> queries(QUERIES);
    std::mt19937 gen(42);
    std::uniform_int_distribution<uint32_t> dist(0, dataset.size() - 1);
    for (auto &query: queries) {
        query = dist(gen);
    }

    timer.Start();

    for (auto repeat = 0; repeat < REPEATS; ++repeat) {
        for (auto i: queries) {
            auto example = dataset.get_example(i);
            auto output = lr.query_probabilities(example);
            sum += output[0] + lr.query_storage(i, example).first;
        }
    }
    nanos = timer.ElapsedNanos(true);
    auto nanosKey = nanos / static_cast<double>(TOT_QUERIES);
    std::cout << "Model inference and retrieval time " << nanos << " ns (" << (nanosKey) << " ns/query)\n";
    benchOutput.push_back("inf_retrieval_nanos=" + std::to_string(nanosKey));
    timer.Start();

    auto start = std::chrono::high_resolution_clock::now();
    for (auto repeat = 0; repeat < REPEATS; ++repeat) {
        for (auto i: queries) {
            auto example = dataset.get_example(i);
            uint64_t res = lr.query(i, example);
            sum += res;
        }
    }
    nanos = timer.ElapsedNanos(true);
    nanosKey = nanos / static_cast<double>(TOT_QUERIES);
    std::cout << "Total query time: " << nanos << " ns (" << nanosKey << " ns/query)\n";
    benchOutput.push_back("query_nanos=" + std::to_string(nanosKey));

    bool ok = true;
    for (size_t i = 0; i < dataset.size(); ++i) {
        auto example = dataset.get_example(i);
        auto label = dataset.get_label(i);
        uint64_t res = lr.query(i, example);
        bool found = res == label;
        assert(found);
        ok &= found;
    }
    if (!ok) {
        std::cerr << "FAILED\n";
        exit(EXIT_FAILURE); // prevent incorrect outputs
    }

    printResult(benchOutput);
}

std::string extraxt(std::string name, std::string prefix) {
    std::string suffix = name.substr(name.find(prefix) + 1, name.length());
    return suffix.substr(0, suffix.find('_'));
}

template<typename Model>
void dispatchStorage(const learnedretrieval::BinaryDatasetReader &dataset, Model &model,
                     std::vector<std::string> benchOutput, std::string modelName,
                     std::filesystem::path modelPath) {
    std::cout << "### Next model: " << modelName << std::endl;


    // model
    auto evalFile = modelPath.string() + "_eval.txt";
    std::ifstream evalStream;
    evalStream.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    evalStream.open(evalFile);
    std::string line;
    std::getline(evalStream, line);
    std::istringstream iss(line);
    std::string token;
    while (iss >> token)
        benchOutput.push_back(token);

    benchOutput.push_back("model_bits=" + std::to_string(8 * model.model_bytes()));
    benchOutput.push_back("model_name=" + modelName);
    benchOutput.push_back("model_l=" + extraxt(modelName, "_L"));
    benchOutput.push_back("model_h=" + extraxt(modelName, "_H"));
    double entropy = 0;
    for (int i = 0; i < dataset.size(); ++i) {
        constexpr float min_prob = std::pow(2.f, -31.f);
        entropy -= std::log2(std::max(model.invoke(dataset.get_example(i))[dataset.get_label(i)], min_prob));
    }
    benchOutput.push_back("cross_entropy_bit_per_key=" + std::to_string(entropy / dataset.size()));

    rocksdb::StopWatchNano timer(true);
    std::vector<uint32_t> queries(QUERIES);
    std::mt19937 gen(42);
    std::uniform_int_distribution<uint32_t> dist(0, dataset.size() - 1);
    for (auto &query: queries) {
        query = dist(gen);
    }
    volatile auto sum = 0;
    timer.Start();
    for (auto repeat = 0; repeat < REPEATS; ++repeat) {
        for (auto i: queries) {
            auto example = dataset.get_example(i);
            auto output = model.invoke(example);
            sum += output[0];
        }
    }
    auto nanos = timer.ElapsedNanos(true);

    std::cout << "Model inference time: " << nanos << " ns ("
              << (nanos / static_cast<double>(TOT_QUERIES))
              << " ns/query)\n";

    benchOutput.push_back("model_inf_ns=" + std::to_string(nanos / static_cast<double>(TOT_QUERIES)));

    // storage
    bool allStorage = storageInput == ALL;

    if (allStorage or storageInput == "filter_huf") {
        bemchmark<learnedretrieval::FilteredRetrievalStorage<learnedretrieval::BitWiseFilterCoding<learnedretrieval::FilterHuffmanCoder>>, Model>(
                dataset, model, benchOutput);
    }
    if (allStorage or storageInput == "filter_fano") {
        bemchmark<learnedretrieval::FilteredRetrievalStorage<learnedretrieval::BitWiseFilterCoding<learnedretrieval::FilterFanoCoder>>, Model>(
                dataset, model, benchOutput);
    }
    if (allStorage or storageInput == "filter_fano50") {
        bemchmark<learnedretrieval::FilteredRetrievalStorage<learnedretrieval::BitWiseFilterCoding<FilteredFano50>>, Model>(
                dataset,
                model,
                benchOutput);
    }
}

void dispatchAllModelsRecurse(const std::string &datasetName, const learnedretrieval::BinaryDatasetReader &dataset,
                              std::vector<std::string> benchOutput, std::string dir) {
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
                    Model model(p);
                    dispatchStorage<Model>(dataset, model, benchOutput, fileName, p);
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
    learnedretrieval::BinaryDatasetReader dataset(rootDir + datasetName);

    std::vector<size_t> cnt;
    cnt.resize(dataset.classes_count());
    for (int i = 0; i < dataset.size(); ++i) {
        cnt[dataset.get_label(i)]++;
    }
    auto sum = double(std::accumulate(cnt.begin(), cnt.end(), 0));
    double entropy = 0;
    for (size_t c: cnt) {
        double p = double(c) / sum;
        entropy -= p * log(p);
    }
    benchOutput.push_back("entropy=" + std::to_string(entropy));
    benchOutput.push_back("size=" + std::to_string(dataset.size()));
    benchOutput.push_back("features=" + std::to_string(dataset.features_count()));
    benchOutput.push_back("classes=" + std::to_string(dataset.classes_count()));

    // model
    dispatchAllModelsRecurse(datasetName, dataset, benchOutput, rootDir);
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
    cmd.add_string('s', "storage", storageInput, "Name of dataset or all");

    if (!cmd.process(argc, argv)) {
        cmd.print_usage();
        return EXIT_FAILURE;
    }

    std::vector<std::string> benchOutput;
    benchOutput.emplace_back("comp=ours");
    dispatchDataSet(benchOutput);
    return EXIT_SUCCESS;
}
