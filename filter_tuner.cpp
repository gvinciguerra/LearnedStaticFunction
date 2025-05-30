#include <cmath>
#include <iostream>

double solveP(int b1, int b2) {
    double eps1 = pow(2.0, -b1);
    double eps2 = pow(2.0, -b2);
    return (eps1 - eps2) / (eps1 - b1 - eps2 + b2);
}

double spaceBinaryFilter(int b, double p) {
    double eps = pow(2.0, -b);
    return p+(1-p)*eps+p * b;
}


int main(int argc, char *argv[]) {
    std::cout<<"{";
    for (int i = 0; i < 10; ++i) {
        std::cout<<solveP(i,i+1)<<", ";
    }
    std::cout<<"};";

    std::cout<<std::endl<<std::endl;

    double p=0.5;
    int b=0;
    while (p>0) {
        while(solveP(b, b+1)>p) {
            b++;
        }
        double spaceOneBit=1.0;
        double spaceEntropy=-p* log2(p) - (1.0-p)* log2(1.0-p);
        double spaceOptFilter=p>0.40938?1.0:p+p/ log(2.0)-p* log2(p/((1.0-p)*log(2.0)));
        double spaceBinFilter = spaceBinaryFilter(b,p);

        std::cout<<"RESULT method=onebit space="<<spaceOneBit<<" overhead="<< (spaceOneBit - spaceEntropy) / spaceEntropy<<" p="<<p<<std::endl;
        std::cout<<"RESULT method=entropy space="<<spaceEntropy<<" overhead="<< (spaceEntropy - spaceEntropy) / spaceEntropy<<" p="<<p<<std::endl;
        std::cout<<"RESULT method=optfilter space="<<spaceOptFilter<<" overhead="<< (spaceOptFilter - spaceEntropy) / spaceEntropy<<" p="<<p<<std::endl;
        std::cout<<"RESULT method=binfilter space="<<spaceBinFilter<<" overhead="<< (spaceBinFilter - spaceEntropy) / spaceEntropy<<" p="<<p<<std::endl;

        p-=0.005;
    }
}