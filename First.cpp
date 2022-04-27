#include <Eigen/Core>
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

    class Random {
    public:
        using Matrix = Sigma::Matrix;
        using Vector = Sigma::Vector;
        Random() = default;
        Random(unsigned long long seed) : rd{seed} {}
        Matrix makeMatrix(int rows, int columns) {
            assert(rows > 0 && "rows must be positive!");
            assert(columns > 0 && "columns must be positive!");
            return norm_gen.template generate<Matrix>(rows, columns, rd);
        }
        Vector makeVector(int rows) {
            assert(rows > 0 && "rows must be positive!");
            return norm_gen.template generate<Matrix>(rows, 1, rd);
        }

    private:
        Eigen::Rand::Vmt19937_64 rd{42};
        Eigen::Rand::NormalGen<float> norm_gen{0, 10};
    };

    class NetLayer {
    public:
        using Matrix = Sigma::Matrix;
        using Vector = Sigma::Vector;
        NetLayer(int rows, int columns)
            : A_(Generator_.makeMatrix(rows, columns)),
              b_(Generator_.makeVector(rows)) {}
        void print_weights() const {
            std::cout << A_ << std::endl;
        }
        void print_bias() const {
            std::cout << b_ << std::endl;
        }
        void shift_A(const Matrix &A) {
            assert(A.cols() == A_.cols() && "dimensions of matrices must be same");
            assert(A.rows() == A_.rows() && "dimensions of matrices must be same");
            A_ -= A;
        }
        void shift_b(const Vector &b) {
            assert(b.rows() == b_.rows() && "dimensions of vectors must be same");
            b_ -= b;
        }
        Vector predict(const Vector &x) const {
            assert(x.size() == A_.cols() && "dimension x must be equal count columns A_!");
            return Sigma::evaluate0(A_ * x + b_);
        }
        Matrix count_grad_A(const Vector &x, const Vector &u) const {
            assert(x.size() == A_.cols() && "dimension x must be equal count columns A_!");
            assert(u.size() == A_.rows() && "dimension u must be equal count rows A_!");
            return Sigma::evaluate1(A_ * x + b_) * u * x.transpose();
        }
        Vector count_grad_b(const Vector &x, const Vector &u) const {
            assert(x.size() == A_.cols() && "dimension x must be equal count columns A_!");
            assert(u.size() == A_.rows() && "dimension u must be equal count rows A_!");
            return Sigma::evaluate1(A_ * x + b_) * u;
        }
        Vector count_grad_x(const Vector &x, const Vector &u) const {
            assert(x.size() == A_.cols() && "dimension x must be equal count columns A_!");
            assert(u.size() == A_.rows() && "dimension u must be equal count rows A_!");
            return (u.transpose() * Sigma::evaluate1(A_ * x + b_) * A_).transpose();
        }

    private:
        static Random Generator_;
        Matrix A_;
        Vector b_;
    };

    Random NetLayer::Generator_;

    class LossFunction {
    public:
        using Vector = Sigma::Vector;
        struct TrainingDatum {
            Vector Input;
            Vector Output;
        };
        using TrainingData = std::vector<TrainingDatum>;
        static float count_loss_function(const TrainingData &data, const std::vector<Vector> &z) {
            assert(!z.empty());
            float result = 0;
            for (int i = 0; i < z.size(); ++i) {
                assert(z[i].size() == data[i].Output.size() && "dimension z[i] must be equal dimension data[i].Output");
                assert(z[i].size() == z[0].size() && "all vectors must have the same dimension");
                result += (z[i] - data[i].Output).dot(z[i] - data[i].Output);
            }
            return result / z.size();
        }
        static Vector count_starting_gradient(const TrainingData &data, const std::vector<Vector> &z) {
            assert(!z.empty());
            Vector gradient = Vector::Zero(z[0].size());
            for (int i = 0; i < z.size(); ++i) {
                assert(z[i].size() == data[i].Output.size() && "dimension z[i] must be equal dimension data[i].Output");
                assert(z[i].size() == z[0].size() && "all vectors must have the same dimension");
                gradient += z[i] - data[i].Output;
            }
            return 2 * gradient / z.size();
        }
    };

    class NeuralNetwork {
    public:
        using Matrix = Eigen::MatrixXf;
        using Vector = Sigma::Vector;
        using TrainingDatum = ML::LossFunction::TrainingDatum;
        using TrainingData = std::vector<TrainingDatum>;
        NeuralNetwork(int dim1, int dim2, int dim3)
            : NL1(dim2, dim1),
              NL2(dim3, dim2) {}
        void train(const TrainingData &data) {
            std::vector<Vector> z1;
            std::vector<Vector> z2;
            Vector u1;
            Vector u2;
            Vector z1_mean = Vector::Zero(z1[0].size());
            Vector x1_mean = Vector::Zero(data[0].Input.size());
            for (int i = 0; i < data.size(); ++i) {
                z1[i] = NL1.predict(data[i].Input);
                z2[i] = NL2.predict(data[i].Input);
            }
            u2 = LF.count_starting_gradient(data, z2);
            for (int i = 0; i < z1.size(); ++i) {
                z1_mean += z1[i];
            }
            z1_mean /= z1.size();
            NL2.shift_A(0.1 * NL2.count_grad_A(z1_mean, u2));
            NL2.shift_b(0.1 * NL2.count_grad_b(z1_mean, u2));
            u1 = NL1.count_grad_x(z1_mean, u2);
            for (int i = 0; i < data.size(); ++i) {
                x1_mean += data[i].Input;
            }
            x1_mean /= data.size();
            NL1.shift_A(0.1 * NL1.count_grad_A(x1_mean, u1));
            NL1.shift_b(0.1 * NL1.count_grad_b(x1_mean, u1));
        };
        Vector predict(const Vector &x) const {
            return NL2.predict(NL1.predict(x));
        };

    private:
        NetLayer NL1;
        NetLayer NL2;
        LossFunction LF;
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
    using Vector = ML::Sigma::Vector;
    ML::NetLayer Layer1(3, 2);
    Vector x(2);
    x << -2, 3;
    Vector u(3);
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
    using Vector = ML::Sigma::Vector;
    using TrainingDatum = ML::LossFunction::TrainingDatum;
    using TrainingData = std::vector<TrainingDatum>;
    TrainingData data;
    Vector x0(5);
    x0 << -2, 3.5, 5, -4, 0;
    Vector y0(4);
    y0 << -1, 2, 1, -2.7;
    Vector x1(5);
    x1 << 8, 8.4, -0.4, 98, -11;
    Vector y1(4);
    y1 << -10, 0, 7.8, 3.7;
    data.resize(2);
    data[0].Input = x0;
    data[0].Output = y0;
    data[1].Input = x1;
    data[1].Output = y1;
    std::vector<Vector> z;
    z.resize(2);
    Vector z0(4);
    Vector z1(4);
    z0 << 6.5, -7, 13, -22;
    z1 << 7, 9, 0, -15;
    z[0] = z0;
    z[1] = z1;
    ML::LossFunction LF;
    std::cout << LF.count_loss_function(data, z) << std::endl;
    std::cout << std::endl;
    std::cout << LF.count_starting_gradient(data, z) << std::endl;
    std::cout << std::endl;
}

void test_neural_network() {
    using Vector = ML::Sigma::Vector;
    using TrainingDatum = ML::LossFunction::TrainingDatum;
    using TrainingData = std::vector<TrainingDatum>;
    TrainingData data;
    Vector x0(5);
    x0 << -2, 3.5, 5, -4, 0;
    Vector y0(4);
    y0 << -1, 2, 1, -2.7;
    Vector x1(5);
    x1 << 8, 8.4, -0.4, 98, -11;
    Vector y1(4);
    y1 << -10, 0, 7.8, 3.7;
    data.resize(2);
    data[0].Input = x0;
    data[0].Output = y0;
    data[1].Input = x1;
    data[1].Output = y1;
    std::vector<Vector> z;
    z.resize(2);
    Vector z0(4);
    Vector z1(4);
    z0 << 6.5, -7, 13, -22;
    z1 << 7, 9, 0, -15;
    z[0] = z0;
    z[1] = z1;
    Vector x(4);
    x << 0, 2, -0.7, -0.9;
    ML::NeuralNetwork NN(4, 3, 2);
    std::cout << NN.predict(x);
}

void test_all() {
    test_net_layer();
    test_back_propagation();
    test_loss_function();
    test_neural_network();
}

int main() {
    test_all();
    return 0;
}
