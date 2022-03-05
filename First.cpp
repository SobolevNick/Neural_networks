#include <iostream>
#include <cmath>
#include <vector>
#include <stdlib.h>
#include <Eigen/Dense>
class sigma { //класс реализует две функции: сигмоиду и производную сигмоиды
public:
    float evaluate0(float x) { //сигмоида
        return 1/(1+exp(-x));
    }
    float evaluate1(float x) { //производная сигмоиды
        return exp(x)/pow((1+exp(x)),2);
    }
};
class building_block { //класс, в котором генерируется матрица A и вектор b 
public: //случайных чисел, а также функция, строящая по данному вектору x, 
    int n = 0; //A, b и sigma() предсказание, и в заключении градиента A, b и x.  
    int m = 0;
    std::vector <float> x = {};
    std::vector <std::vector<float>> A;
    std::vector <float> b = {};
    std::vector <float> u = {};
    building_block(std::vector <float> y, std::vector <float> z) {//генерирует A и b
        x = y;
        u = z;
        n = y.size();
        m = z.size();
        A = std::vector<std::vector<float>> (n, std::vector<float>(m, 0)) ;
        b = std::vector <float> (m, 0);
        for (int k = 0; k < m; k++) {
            for (int j = 0; j < n; j++) {
                A[k][j] = -1 + rand() % 3; //я пока что зарандомил из трёх чисел 
            }
        }
        for (int i = 0; i < m; i++) {
            b[i] = -1 + rand() % 3;
        }
    }
    std::vector <float> theta() {//вычисляет Ax+b
        std::vector <float> f(m);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++){
                f[i] = f[i] + A[i][j] * x[j];
            }
            f[i] = f[i] + b[i];
        }
        return f;
    }
    std::vector <float> prediction() {//функция предсказания
        sigma sigma0;
        std::vector <float> f = theta();

        for (int i = 0; i < m; i++) {
            f[i] = sigma0.evaluate0(f[i]);
        }
        return f;
    }
    std::vector <std::vector<float>> sigma_der() {//вычисляет диагональную матрицу, 
        std::vector <float> f = theta();//на диагонали которой стоят производные sigma()
        std::vector <std::vector<float>> sd(m, std::vector<float>(m,0));//от Ax+b
        sigma sigma1;
        for (int i = 0; i < m; i++) {
            sd[i][i] = sigma1.evaluate1(f[i]);
        }
        return sd;
    };
    std::vector <float> grad_b() {//градиент b
        std::vector <std::vector<float>> f = sigma_der();
        sigma sigma1;
        std::vector <float> gb(m,0);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
               gb[i] = gb[i] + f[i][j]*u[j];
            }
        }
        return gb;
    }
};

int main() {
    return 0;
}
