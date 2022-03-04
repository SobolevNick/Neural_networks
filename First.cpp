#include <iostream>
#include <cmath>
#include <vector>
#include <stdlib.h>

class sigma {
public:
    float evaluate0(float x) {
        return 1/(1+exp(-x));
    }
    float evaluate1(float x) {
        return exp(x)/pow((1+exp(x)),2);
    }
};

class building_block {
public:
    int n = 0;
    std::vector <float> x = {};
    //float A[0][0];
    building_block(std::vector <float> y) {
        x = y;
        n = y.size();
        float A[n][n];
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < n; j++) {
                A[k][j] = rand();            
            }
        }
    }
    std::vector <float> prediction() {
        sigma sigma0;
        std::vector <float> y(n);
        for (int i = 0; i < n; i++) {
           y[i] = sigma0.evaluate0(x[i]);
        }
        return y;
    }
};

int main() {
    return 0;
}
