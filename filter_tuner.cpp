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

double solveP(int b1, int b2) {
    double eps1 = pow(2.0, -b1);
    double eps2 = pow(2.0, -b2);
    return (eps1 - eps2) / (eps1 - b1 - eps2 + b2);
}

double spaceBinaryFilter(int b, double p) {
    double eps = pow(2.0, -b);
    return p + (1 - p) * eps + p * b;
}

double spaceBinaryFilter0(int b) {
    return 1 + b;
}

double spaceBinaryFilter1(int b) {
    double eps = pow(2.0, -b);
    return eps;
}


int main(int argc, char *argv[]) {
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
        double surprisal0= -log2(p);
        double surprisal1 = -log2(1.0 - p);
        double spaceEntropy = -p * log2(p) - (1.0 - p) * log2(1.0 - p);
        double spaceOptFilter = p > 0.40938 ? 1.0 : p + p / log(2.0) - p * log2(p / ((1.0 - p) * log(2.0)));
        double spaceBinFilter = spaceBinaryFilter(b, p);
        double spaceBin0 = spaceBinaryFilter0(b);
        double spaceBin1 = spaceBinaryFilter1(b);

        std::cout << "RESULT method=onebit space=" << spaceOneBit << " overhead="
                  << (spaceOneBit - spaceEntropy) / spaceEntropy << " p=" << p << std::endl;
        std::cout << "RESULT method=entropy space=" << spaceEntropy << " overhead="
                  << (spaceEntropy - spaceEntropy) / spaceEntropy << " p=" << p << std::endl;
        std::cout << "RESULT method=optfilter space=" << spaceOptFilter << " overhead="
                  << (spaceOptFilter - spaceEntropy) / spaceEntropy << " p=" << p << std::endl;
        std::cout << "RESULT method=binfilter space=" << spaceBinFilter << " overhead="
                  << (spaceBinFilter - spaceEntropy) / spaceEntropy << " p=" << p << std::endl;
        std::cout << "RESULT method=binfilterunlikely space=" << spaceBin0 << " overhead="
                  << (spaceBin0 - surprisal0) / surprisal0 << " p=" << p << std::endl;
        std::cout << "RESULT method=binfilterlikely space=" << spaceBin1 << " overhead="
                  << (spaceBin1 - surprisal1) / surprisal1 << " p=" << p << std::endl;

        p -= 0.001;
    }
}