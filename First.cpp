#include <Eigen/Core>
#include <Eigen/Dense>
#include <EigenRand/EigenRand>
#include <cmath>
#include <iostream>
#include <vector>

namespace ML {
    class Sigma {
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
        Eigen::Rand::Vmt19937_64 urng{42};
        Eigen::Rand::NormalGen<float> norm_gen{0, 10};
        NetLayer(int m, int n)
            : A_(norm_gen.template generate<Matrix>(m, n, urng)), b_(norm_gen.template generate<Matrix>(m, 1, urng)) {
        }
        void print_weights() const {
            std::cout << A_ << std::endl;
        }
        void print_bias() const {
            std::cout << b_ << std::endl;
        }
        Vector predict(const Vector &x) const {
            return Sigma::evaluate0(A_ * x + b_);
        }
        Matrix count_grad_A(const Vector &x, const Vector &u) const {
            return Sigma::evaluate1(A_ * x + b_) * u * x.transpose();
        }
        Vector count_grad_b(const Vector &x, const Vector &u) const {
            return Sigma::evaluate1(A_ * x + b_) * u;
        }
        Vector count_grad_x(const Vector &x, const Vector &u) const {
            return (u.transpose() * Sigma::evaluate1(A_ * x + b_) * A_).transpose();
        }

    private:
        Matrix A_;
        Vector b_;
    };

    class LossFunction {
    public:
        using Vector = Eigen::VectorXf;
        static float count_loss_function(std::vector<std::pair<Eigen::VectorXf, Eigen::VectorXf>> x) {
            NetLayer Layer(x[0].first.size(), x[0].second.size());
            float sum = 0;
            for (int i = 0; i < x.size(); i++) {
                sum = (Layer.predict(x[i].first) - x[i].second).dot(Layer.predict(x[i].first) - x[i].second) + sum;
            }
            return sum / x.size();
        }
        static Vector count_starting_gradient(std::vector<std::pair<Eigen::VectorXf, Eigen::VectorXf>> x) {
            NetLayer Layer(x[0].first.size(), x[0].second.size());
            Vector sum(x[0].second.size());
            sum = Vector::Zero(x[0].second.size());
            for (int i = 0; i < x.size(); i++) {
                sum = Layer.predict(x[i].first) - x[i].second + sum;
            }
            return 2 * sum / x.size();
        }
    };

    class NeuralNetwork {
        using Matrix = Eigen::MatrixXf;
        using Vector = Eigen::VectorXf;
        NeuralNetwork(Vector x, Vector y) {
            NetLayer Layer((y.rows()), (x.rows()));
        }
        static std::pair<Matrix, Vector> count_theta_2(Matrix A, Vector b) {

            return std::pair<Matrix, Vector>(A, b);
        };

    private:
        Vector z_1;
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
    std::cout << Layer1.count_grad_x(x, u) << std::endl
              << std::endl;
}

void test_loss_function() {
    Eigen::VectorXf x(5);
    x << -2, 3.5, 5, -4, 0;
    Eigen::VectorXf y(4);
    y << -1, 2, 1, -2.7;
    ML::LossFunction LF;
    //std::cout << LF.count_loss_function(x, y);
    //std::cout << LF.count_starting_gradient_i(x, y);
}

void test_neural_network() {
    Eigen::VectorXf x(5);
    x << -2, 3.5, 5, -4, 0;
    Eigen::VectorXf y(4);
    y << -1, 2, 1, -2.7;
    //ML::NeuralNetwork NN;
    //std::cout << NN.count_theta_2(x, y);
    //std::cout << NN.count_starting_gradient(x, y);
}

void test_all() {
    test_net_layer();
    test_back_propagation();
    test_loss_function();
}

int main() {
    test_all();
}
