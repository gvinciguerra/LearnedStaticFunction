#include <cmath>
#include <iostream>

#include <algorithm>
#include <cassert>
#include <vector>
#include <ranges>
#include <numeric>
#include <queue>
#include <functional>
#include <cstdint>
#include <bitset>
#include <span>


template<typename Symbol = uint32_t, typename Frequency = float>
class FilterFanoCoder {
    struct Elem {
        Symbol s;
        Frequency p;
    };

    std::vector<Elem> sorted;
    static constexpr int BUCKETS = 20;
    std::array<size_t, BUCKETS> bucketCnt = {};

    int getBucket(Frequency f) {
        int exp;
        frexp(f, &exp);
        return std::min(1-exp, BUCKETS - 1);
    }

public:

    FilterFanoCoder(const std::vector<Frequency> &f) {
        for (const Frequency e: f) {
            bucketCnt[getBucket(e)]++;
        }
        size_t sum = 0;
        for (size_t i = 0; i < BUCKETS; ++i) {
            size_t offset = sum;
            sum += bucketCnt[i];
            bucketCnt[i] = offset;
        }
        sorted.reserve(f.size());
        for (Symbol i = 0; i < f.size(); ++i) {
            size_t pos = bucketCnt[getBucket(f[i])]++;
            sorted[pos] = {i, f[i]};
        }
        for (Symbol i = 0; i < f.size(); ++i) {
            std::cout<<sorted[i].p<<std::endl;
        }
    }


    float getRelProbability() {
    }

    bool hasFinished() {
    }

    void nextBit(bool bit) {
    }

    Symbol getResult() {
    }


};


double solveP(int b1, int b2) {
    double eps1 = pow(2.0, -b1);
    double eps2 = pow(2.0, -b2);
    return (eps1 - eps2) / (eps1 - b1 - eps2 + b2);
}

double spaceBinaryFilter(int b, double p) {
    double eps = pow(2.0, -b);
    return p + (1 - p) * eps + p * b;
}


int main(int argc, char *argv[]) {
    std::vector<float> f = {2.10818e-07, 0.00172702, 0.884235, 5.39694e-05, 0.00345404, 0.110529, 1.64702e-09};
    std::cout << std::reduce(f.begin(), f.end()) << std::endl;
    auto coder = FilterFanoCoder(f);
    return 0;
    /*for (std::pair<float, bool> p: coder.getProbabilities(5)) {
        std::cout << p.first << " " << p.second << std::endl;
    }*/

    std::cout << std::endl << std::endl;

    auto coder2 = FilterFanoCoder(f);

    std::cout << coder2.getRelProbability() << " " << coder2.hasFinished() << std::endl;
    coder2.nextBit(0);
    std::cout << coder2.getRelProbability() << " " << coder2.hasFinished() << std::endl;
    coder2.nextBit(1);
    std::cout << coder2.getRelProbability() << " " << coder2.hasFinished() << std::endl;
    coder2.nextBit(1);
    std::cout << coder2.getRelProbability() << " " << coder2.hasFinished() << std::endl;
    std::cout << coder2.getResult();


    return 0;


    std::cout << "{";
    for (int i = 0; i < 30; ++i) {
        std::cout << solveP(i, i + 1) << ", ";
    }
    std::cout << "};";

    std::cout << std::endl << std::endl;

    double p = 0.5;
    int b = 0;
    while (p > 0) {
        while (solveP(b, b + 1) > p) {
            b++;
        }
        double spaceOneBit = 1.0;
        double spaceEntropy = -p * log2(p) - (1.0 - p) * log2(1.0 - p);
        double spaceOptFilter = p > 0.40938 ? 1.0 : p + p / log(2.0) - p * log2(p / ((1.0 - p) * log(2.0)));
        double spaceBinFilter = spaceBinaryFilter(b, p);

        std::cout << "RESULT method=onebit space=" << spaceOneBit << " overhead="
                  << (spaceOneBit - spaceEntropy) / spaceEntropy << " p=" << p << std::endl;
        std::cout << "RESULT method=entropy space=" << spaceEntropy << " overhead="
                  << (spaceEntropy - spaceEntropy) / spaceEntropy << " p=" << p << std::endl;
        std::cout << "RESULT method=optfilter space=" << spaceOptFilter << " overhead="
                  << (spaceOptFilter - spaceEntropy) / spaceEntropy << " p=" << p << std::endl;
        std::cout << "RESULT method=binfilter space=" << spaceBinFilter << " overhead="
                  << (spaceBinFilter - spaceEntropy) / spaceEntropy << " p=" << p << std::endl;

        p -= 0.005;
    }
}