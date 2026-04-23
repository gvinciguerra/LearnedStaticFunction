#define QUERIES 10000000
#define REPEATS 10

extern "C" {
#define restrict __restrict__
#include "c/spooky.h"
#ifdef CSF3
#include "c/csf3.h"
#define CSF_GET csf3_get_byte_array
#define CSF_EXT ".csf3"
#else
#include "c/csf4.h"
#define CSF_GET csf4_get_byte_array
#define CSF_EXT ".csf4"
#endif
}

#include <iostream>
#include <vector>
#include <tuple>
#include <fstream>
#include <string>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <random>

std::tuple<size_t, size_t, std::string> read_file(const std::string& filename) {
    std::ifstream file(filename);
    std::string inputs;
    std::string line;
    size_t input_bytes = 0;
    size_t n = 0;
    if (file.is_open()) {
        file.seekg(0, std::ios::end);
        std::streampos fileSize = file.tellg();
        file.seekg(0, std::ios::beg);
        inputs.reserve(fileSize);
        std::getline(file, line);
        inputs.append(line);
        input_bytes = line.size();
        n = 1;
        while (std::getline(file, line)) {
            inputs.append(line);
            ++n;
        }
    } else {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(1);
    }
    return std::make_tuple(n, input_bytes, std::move(inputs));
}

std::vector<int64_t> read_int64_binary_big_endian(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    std::vector<int64_t> numbers;
    if (file.is_open()) {
        file.seekg(0, std::ios::end);
        std::streampos fileSize = file.tellg();
        file.seekg(0, std::ios::beg);
        std::size_t numValues = fileSize / sizeof(uint64_t);
        numbers.resize(numValues);
        file.read(reinterpret_cast<char*>(numbers.data()), fileSize);
        for (auto &x : numbers) {
            x = __builtin_bswap64(x);
        }
    } else {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(1);
    }
    return numbers;
}


int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <dataset_path> <csf_path>" << std::endl;
        exit(1);
    }

    std::string dataset_path = argv[1];
    std::string csf_path = argv[2];
    auto [n, input_bytes, inputs] = read_file(dataset_path + "_X.sux4j");
    auto outputs = read_int64_binary_big_endian(dataset_path + "_y.sux4j");
    if (n != outputs.size()) {
        std::cerr << "Input and output sizes do not match" << std::endl;
        exit(1);
    }

    int h = open(csf_path.c_str(), O_RDONLY);
    if (h < 0) {
        std::cerr << "Failed to open file: " << csf_path << std::endl;
        exit(1);
    }
    csf *csf = load_csf(h);
	close(h);

    auto query = [&](const uint32_t i) { return CSF_GET(csf, inputs.data() + i * input_bytes, input_bytes); };

    for (uint32_t i = 0; i < n; i++) {
        if (query(i) != outputs[i]) {
            std::cerr << "Mismatch at index: " << i << std::endl;
            std::cerr << "Input: " << std::string_view(inputs.data() + i * input_bytes, input_bytes) << std::endl;
            std::cerr << "Expected: " << outputs[i] << std::endl;
            std::cerr << "Got: " << query(i) << std::endl;
            exit(1);
        }
    }

    std::vector<uint32_t> queries(QUERIES);
    std::mt19937 gen(42);
    std::uniform_int_distribution<uint32_t> dist(0, n - 1);
    for (auto &query : queries) {
        query = dist(gen);
    }

    uint64_t signature[4];
    auto start_hash = std::chrono::high_resolution_clock::now();
    for (auto repeat = 0; repeat < REPEATS; ++repeat) {
        for (auto i: queries) {
            spooky_short(inputs.data() + i * input_bytes, input_bytes, csf->global_seed, signature);
        }
    }
    auto end_hash = std::chrono::high_resolution_clock::now();
    auto duration_hash = std::chrono::duration_cast<std::chrono::nanoseconds>(end_hash - start_hash).count();

    size_t sum = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (auto repeat = 0; repeat < REPEATS; ++repeat) {
        for (auto i: queries) {
            sum += query(i);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    std::cout << "Compressed function file: " << csf_path << std::endl;
    std::cout << "Results sum: " << sum << std::endl;
    std::cout << "Time: " << duration / (queries.size() * REPEATS)  << " ns/query" << std::endl;
    std::cout << "Hash time: " << duration_hash / (queries.size() * REPEATS)  << " ns/query" << std::endl;
    std::cout << "Size: " << n << std::endl;

    return 0;
}