#include <Eigen/Dense>
#include <cmath>
#include <iostream>

namespace ML {
    class sigma {
    public:
        using Matrix = Eigen::MatrixXf;
        using Vector = Eigen::VectorXf;
        static float evaluate0(float x) {
            return 1 / (1 + exp(-x));
        }
        static float evaluate1(float x) {
            return exp(x) / pow((1 + exp(x)), 2);
        };
        static Vector evaluate0(const Vector &x) {
            return x.array().exp() / (1 + x.array().exp());
        };
        static Matrix evaluate1(const Vector &x) {
            return (x.array().exp() / pow(1 + x.array().exp(), 2)).matrix().asDiagonal();
        };
    };

    class NetLayer {
    public:
        using Matrix = Eigen::MatrixXf;
        using Vector = Eigen::VectorXf;
        NetLayer(int m, int n)
            : A_(Matrix::Random(m, n)), b_(Vector::Random(m)) {
        }
        void print_weights() const {
            std::cout << A_ << std::endl;
        };
        void print_bias() const {
            std::cout << b_ << std::endl;
        };
        Vector predict(const Vector &x) const {
            return sigma::evaluate0(A_ * x + b_);
        }
        Matrix count_grad_A(Vector x, Vector u) const {
            return sigma::evaluate1(A_ * x + b_) * u * x.transpose();
        }
        Vector count_grad_b(Vector x, Vector u) const {
            return sigma::evaluate1(A_ * x + b_) * u;
        }
        Vector count_grad_x(Vector x, Vector u) const {
            return (u.transpose() * sigma::evaluate1(A_ * x + b_) * A_).transpose();
        }

    private:
        int in_size() const {//Не использовалась пока
            return A_.cols();
        }
        int out_size() const {
            return A_.rows();
        }
        Matrix A_;
        Vector b_;
    };
}// namespace ML
void test_net_layer() {
    ML::NetLayer Layer1(30, 20);
    Layer1.print_weights();
    std::cout << std::endl;
    ML::NetLayer Layer2(25, 35);
    Layer2.print_bias();
    std::cout << std::endl;
}
void test_back_propagation() {
    ML::NetLayer Layer1(3, 2);
    Eigen::VectorXf x(2);
    x << -2, 3;
    Eigen::VectorXf u(3);
    u << -1, 2, 1;
    std::cout << Layer1.predict(x) << std::endl
              << std::endl;
    std::cout << Layer1.count_grad_A(x, u) << std::endl
              << std::endl;
    std::cout << Layer1.count_grad_b(x, u) << std::endl
              << std::endl;
    std::cout << Layer1.count_grad_x(x, u) << std::endl;
}
void test_all() {
    test_net_layer();
    test_back_propagation();
}
int main() {
    test_all();
}

