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

#include "learnedretrieval/learned_static_function.hpp"
#include "learnedretrieval/model_gauss.hpp"

#define QUERIES 10000000
#define REPEATS 10
#define TOT_QUERIES (QUERIES*REPEATS)

std::string rootDir = "lrdata/";
constexpr std::string ALL = "all";
std::string modelInput = ALL;
std::string dataSetInput = ALL;
std::string storageInput = ALL;


void printResult(const std::vector<std::string> &benchOutput) {
    std::cout << std::endl << "RESULT ";
    for (auto s: benchOutput) {
        std::cout << s << " ";
    }
    std::cout << std::endl;
};


template<typename S, typename F>
class FilteredFano50 : public lsf::Filter50PercentWrapper<lsf::FilterFanoCoder, S, F> {
};

template<typename Storage, typename Model>
void
benchmark(const lsf::BinaryDatasetReader &dataset, Model &model, std::vector<std::string> benchOutput) {

    std::cout << "### Next storage: " << Storage::get_name() << std::endl;

    benchOutput.emplace_back("comp=ours");
    benchOutput.push_back("storage_name=" + Storage::get_name());
    rocksdb::StopWatchNano timer(true);

    lsf::LearnedStaticFunction<Model, Storage> lr(dataset, model);

    auto nanos = timer.ElapsedNanos(true);
    std::cout << "Total Construct " << nanos << " ns ("
              << (nanos / static_cast<double>(dataset.size())) << " ns/key)\n";

    benchOutput.push_back("construct_ms=" + std::to_string(double(nanos) / 1000.0 / 1000.0));
    benchOutput.push_back("storage_bits=" + std::to_string(8.0 * lr.storage_bytes() / double(dataset.size())));
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


template<size_t r>
void benchmarkBuRR(const lsf::BinaryDatasetReader &dataset, std::vector<std::string> benchOutput) {

    std::cout << "### Next storage: BuRR" << std::endl;

    benchOutput.emplace_back("comp=BuRR");
    benchOutput.emplace_back("storage_name=BuRR");
    rocksdb::StopWatchNano timer(true);

    using Config = ribbon::FastRetrievalConfig<r, uint64_t>;
    using RibbonT = ribbon::ribbon_filter<2, Config>;
    RibbonT retrievalDs(dataset.size(), 0.965, 42);
    std::vector<std::pair<uint64_t, uint8_t>> data;
    for (int i = 0; i < dataset.size(); ++i) {
        data.emplace_back(i, dataset.get_label(i));
    }
    retrievalDs.AddRange(data.begin(), data.end());
    retrievalDs.BackSubst();

    auto nanos = timer.ElapsedNanos(true);
    std::cout << "Total Construct " << nanos << " ns ("
              << (nanos / static_cast<double>(dataset.size())) << " ns/key)\n";

    benchOutput.push_back("construct_ms=" + std::to_string(double(nanos) / 1000.0 / 1000.0));
    benchOutput.push_back("storage_bits=" + std::to_string(8.0 * retrievalDs.Size() / double(dataset.size())));
    volatile uint64_t sum = 0;

    std::vector<uint32_t> queries(QUERIES);
    std::mt19937 gen(42);
    std::uniform_int_distribution<uint32_t> dist(0, dataset.size() - 1);
    for (auto &query: queries) {
        query = dist(gen);
    }

    timer.Start();

    auto start = std::chrono::high_resolution_clock::now();
    for (auto repeat = 0; repeat < REPEATS; ++repeat) {
        for (auto i: queries) {
            uint64_t res = retrievalDs.QueryRetrieval(i);
            sum += res;
        }
    }
    nanos = timer.ElapsedNanos(true);
    auto nanosKey = nanos / static_cast<double>(TOT_QUERIES);
    std::cout << "Total query time: " << nanos << " ns (" << nanosKey << " ns/query)\n";
    benchOutput.push_back("query_nanos=" + std::to_string(nanosKey));

    bool ok = true;
    for (size_t i = 0; i < dataset.size(); ++i) {
        auto label = dataset.get_label(i);
        uint64_t res = retrievalDs.QueryRetrieval(i);
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

static inline uint64_t lg_down(uint64_t x) {
    return 63U - __builtin_clzl(x);
}

/* base-2 logarithm, rounding up */
static inline uint64_t lg_up(uint64_t x) {
    return lg_down(x - 1) + 1;
}

template<size_t r = 1>
void
dispatchBuRR(const lsf::BinaryDatasetReader &dataset, std::vector<std::string> benchOutput) {
    size_t target = lg_up(dataset.classes_count());
    if (target == r) {
        benchmarkBuRR<r>(dataset, benchOutput);
    } else {
        if constexpr (r < 10) {
            dispatchBuRR<r + 1>(dataset, benchOutput);
        } else {
            std::cerr << "Too many classes, not compiled" << std::endl;
        }
    }
}

template<typename Model>
void dispatchStorage(const lsf::BinaryDatasetReader &dataset, Model &model,
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
        benchmark<lsf::FilteredLSFStorage<lsf::BitWiseFilterCoding<lsf::FilterHuffmanCoder>>, Model>(
                dataset, model, benchOutput);
    }
    if (allStorage or storageInput == "filter_fano") {
        benchmark<lsf::FilteredLSFStorage<lsf::BitWiseFilterCoding<lsf::FilterFanoCoder>>, Model>(
                dataset, model, benchOutput);
    }
    if (allStorage or storageInput == "filter_fano50") {
        benchmark<lsf::FilteredLSFStorage<lsf::BitWiseFilterCoding<FilteredFano50>>, Model>(
                dataset,
                model,
                benchOutput);
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
                    dispatchStorage<lsf::ModelWrapper>(dataset, model, benchOutputCopy, fileName);
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


    // storages that dont need model
    if (storageInput == ALL or storageInput == "BuRR") {
        dispatchBuRR(dataset, benchOutput);
    }

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
        dispatchStorage<lsf::ModelGaussianNaiveBayes>(dataset, model, benchOutput, "gauss");
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
    cmd.add_string('s', "storage", storageInput, "Name of dataset or all");

    if (!cmd.process(argc, argv)) {
        cmd.print_usage();
        return EXIT_FAILURE;
    }

    std::vector<std::string> benchOutput;
    dispatchDataSet(benchOutput);
    return EXIT_SUCCESS;
}
