#include <iostream>
#include <cmath>
#include <Eigen/Dense>

namespace ML {
class sigma {
public:
  static float evaluate0(float x) {
    return 1 / (1 + exp(-x));
  }
  static float evaluate1(float x) {
    return exp(x) / pow((1 + exp(x)),2);
  };
  static Eigen::VectorXf evaluate0(const Eigen::VectorXf& x) {
    return x.array().exp() / (1 + x.array().exp());
  };
  //static Eigen::MatrixXf evaluate1(const Eigen::VectorXf& x) {
    //return Eigen::MatrixXf::Identity().array() * (x.array().exp() / pow(1 + x.array().exp(), 2));
  //};
};

class NetLayer {
public:
  using Matrix = Eigen::MatrixXf;
  using Vector = Eigen::VectorXf;
  using RowVector = Eigen::RowVectorXf;
  NetLayer(int m, int n)
      : A_(Matrix::Random(m,n)), b_(Vector::Random(m)) {
  }
  void print_weights() const {
    std::cout << A_ << std::endl;
  };
  void print_bias() const {
    std::cout << b_ << std::endl;
  };
  Vector predict(const Vector& x) const {
    return sigma::evaluate0(A_ * x + b_);
  }
  Matrix count_grad_A(Vector x, Vector u) const {
    Matrix S(out_size(),out_size());
    for (int i = 0; i < out_size(); i++) {
      S(i,i) = sigma::evaluate1((A_ * x + b_)(i));
    }
    return S * u * x.transpose();
  }
  Vector count_grad_b(Vector x, Vector u) const {
    Matrix S(out_size(),out_size());
    for (int i = 0; i < out_size(); i++) {
      S(i,i) = sigma::evaluate1((A_ * x + b_)(i));
    }
    return S * u;
  }
  RowVector count_grad_x(Vector x, Vector u) const {
    Matrix S(out_size(),out_size());
    for (int i = 0; i < out_size(); i++) {
      S(i,i) = sigma::evaluate1((A_ * x + b_)(i));
    }
    return u.transpose() * S * A_;
  }
private:
  int in_size() const { //Не использовалась пока 
    return A_.cols();
  }
  int out_size() const {
    return A_.rows();
  }
  Matrix A_;
  Vector b_;
};
}
void test_net_layer() {
  ML::NetLayer print_weights(); //как-то не работает 
  ML::NetLayer print_bias();
}
void test_back_propagation() {

}
void test_all() {
  test_net_layer();
  test_back_propagation();
}
int main() {
  test_all();
}

