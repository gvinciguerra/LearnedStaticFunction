#include <atomic>
#include <cstdlib>
#include <numeric>
#include <thread>
#include <vector>
#include <iostream>
#include <chrono>

#include "ribbon.hpp"
#include "serialization.hpp"
#include "rocksdb/stop_watch.h"

#include <tlx/cmdline_parser.hpp>
#include <tlx/logger.hpp>

#include "learnedretrieval/learned_retrieval.hpp"


int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <dataset_path> <model_path>" << std::endl;
        return 1;
    }

    auto dataset_path = argv[1];
    auto model_path = argv[2];

    learnedretrieval::BinaryDatasetReader dataset(dataset_path);

    std::cout << "Dataset:" << std::endl
              << "  Examples: " << dataset.size() << std::endl
              << "  Features: " << dataset.features_count() << std::endl
              << "  Classes:  " << dataset.classes_count() << std::endl;

    learnedretrieval::LearnedRetrieval<learnedretrieval::FilteredRetrievalStorage> lr(dataset, model_path);
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

    return 0;
}
