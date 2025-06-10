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
        Frequency f;
        Symbol s;
        uint64_t cumP;

    public:
        bool getCumBit(size_t pos) {
            return (cumP >> pos) & 1;
        }
    };

    class Compare {
    public:
        bool operator()(Elem a, Elem b) {
            return a.f < b.f;
        }
    };

    size_t n;
    std::priority_queue<Elem, std::vector<Elem>, Compare> in;
    std::vector<Elem> sorted;

    size_t leftBound;
    size_t rightBound;
    size_t center;
    Frequency lastCumFreq;
    static constexpr size_t HIGHEST_BIT = 63;
    size_t currentBitPos = HIGHEST_BIT - 1;
    static constexpr uint64_t P_ONE_INT = uint64_t(1)<<HIGHEST_BIT;
    static constexpr float P_ONE = float(P_ONE_INT);
    float lastCumP = 0;
    uint64_t lastCumPint = -1;

    Elem getElem(size_t i) {
        while (sorted.size() <= i) {
            Elem e = in.top();
            lastCumPint = std::max<uint64_t>(lastCumPint + 1, std::min<uint64_t>(uint64_t(lastCumP * P_ONE), P_ONE_INT - n + sorted.size())); // make sure all codes are unique
            e.cumP = lastCumPint;
            std::cout << e.s << " " << std::bitset<64>(e.cumP).to_string() << " " << e.f << std::endl;
            lastCumP += e.f;
            sorted.push_back(e);
            in.pop();
        }
        return sorted[i];
    }

public:

    FilterFanoCoder(const std::vector<Frequency> &f) {
        n=f.size();
        for (Symbol i = 0; i < f.size(); ++i) {
            in.push({f[i], i, 0});
        }
        lastCumFreq = 1.0;
        leftBound = 0;
        rightBound = f.size() - 1;
    }

    std::vector<std::pair<float, bool>> getProbabilities(Symbol s) {
        // find the element
        Elem target;
        int i = 0;
        while ((target = getElem(i++)).s != s);


        std::vector<std::pair<float, bool>> res;
        while (leftBound != rightBound) {
            float absoluteFreq = 0;
            int index = leftBound;
            while (true) {
                if (index == rightBound + 1) { // all 0 skip
                    absoluteFreq = 0;
                    index = leftBound;
                    std::cout<<"HEREA "<<currentBitPos<<std::endl;
                    currentBitPos--;
                }
                Elem e = getElem(index);
                if (e.getCumBit(currentBitPos)) {
                    if(index==leftBound) { // all 1 skip
                        absoluteFreq = 0;
                        std::cout<<"HEREB "<<currentBitPos<<std::endl;
                        currentBitPos--;
                        continue;
                    } else {
                        break;
                    }
                }
                absoluteFreq += e.f;
                index++;
            }
            center = index; // center points to first element with 1 bit in current pos
            float currentRelFeq = absoluteFreq / lastCumFreq;
            lastCumFreq = absoluteFreq;

            bool bit = target.getCumBit(currentBitPos);
            if (bit) {
                leftBound = center;
            } else {
                rightBound = center - 1;
            }
            std::cout<<"HEREC "<<currentBitPos<<std::endl;
            currentBitPos--;

            res.push_back({currentRelFeq, bit});
        }
        return res;
    }

    float getRelProbability() {
        float absoluteFreq = 0;
        int index = leftBound;
        while (true) {
            Elem e = getElem(index);
            if (e.getCumBit(currentBitPos)) {
                break;
            }
            absoluteFreq += e.f;
            index++;
        }
        center = index;
        float currentRelFeq = absoluteFreq / lastCumFreq;
        lastCumFreq = absoluteFreq;
        return currentRelFeq;
    }

    bool hasFinished() {
        return leftBound == rightBound;
    }

    void nextBit(bool bit) {
        if (bit) {
            leftBound = center;
        } else {
            rightBound = center - 1;
        }

        currentBitPos--;
    }

    Symbol getResult() {
        return getElem(leftBound).s;
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
    std::vector<float> f = {2.10818e-07,0.00172702,0.884235,5.39694e-05,0.00345404,0.110529,1.64702e-09};
    std::cout << std::reduce(f.begin(), f.end()) << std::endl;
    auto coder = FilterFanoCoder(f);
    for (std::pair<float, bool> p: coder.getProbabilities(5)) {
        std::cout << p.first << " " << p.second << std::endl;
    }

    std::cout<<std::endl<<std::endl;

    auto coder2 = FilterFanoCoder(f);

    std::cout<< coder2.getRelProbability()<<" "<<coder2.hasFinished()<<std::endl;
    coder2.nextBit(0);
    std::cout<< coder2.getRelProbability()<<" "<<coder2.hasFinished()<<std::endl;
    coder2.nextBit(1);
    std::cout<< coder2.getRelProbability()<<" "<<coder2.hasFinished()<<std::endl;
    coder2.nextBit(1);
    std::cout<< coder2.getRelProbability()<<" "<<coder2.hasFinished()<<std::endl;
    std::cout<<coder2.getResult();



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