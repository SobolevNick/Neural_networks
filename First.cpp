#include <iostream>
#include <cmath>
#include <Eigen/Dense>

class sigma {
public:
    float evaluate0(float x) {
        return 1/(1+exp(-x));
    }
    float evaluate1(float x) {
        return exp(x)/pow((1+exp(x)),2);
    }
};
class theta {
    public:
    Eigen::VectorXf count_theta(Eigen::MatrixXf A, Eigen::VectorXf b, Eigen::VectorXf x) {
        return A*x + b;
    }
};
class building_block {
private:
    Eigen::MatrixXf A_;
    Eigen::VectorXf b_;
public:
    building_block(int m, int n) {
        A_ = Eigen::MatrixXf::Random(m,n);
        b_ = Eigen::VectorXf::Random(m);
        }
    void check_mat() {
        std::cout << A_ << std::endl;
    };
    void check_vec () {
        std::cout << b_ << std::endl;
    };
    Eigen::VectorXf predict(Eigen::VectorXf x) {
        sigma s;
        theta t;
        int m = t.count_theta(A_,b_,x).size();
        Eigen::VectorXf pred(m);
        for (int i = 0; i < m; i++)  {
            pred(i) = s.evaluate0(t.count_theta(A_,b_,x)(i));
        }
        return pred;
    }
    Eigen::MatrixXf count_grad_A(Eigen::VectorXf x, Eigen::RowVectorXf u) {
        theta t;
        sigma s;
        int m = u.size();
        Eigen::MatrixXf S(m,m);
        for (int i = 0; i < m; i++) {
            S(i,i) = s.evaluate1(t.count_theta(A_,b_,x)(i));
        }
        return S*u.transpose()*x.transpose();
    }
    Eigen::VectorXf count_grad_b(Eigen::VectorXf x, Eigen::RowVectorXf u) {
        theta t;
        sigma s;
        int m = u.size();
        Eigen::MatrixXf S(m,m);
        for (int i = 0; i < m; i++) {
            S(i,i) = s.evaluate1(t.count_theta(A_,b_,x)(i));
        }
        return S*u.transpose();
    }
    Eigen::RowVectorXf count_grad_x(Eigen::VectorXf x, Eigen::RowVectorXf u) {
        theta t;
        sigma s;
        int m = u.size();
        Eigen::MatrixXf S(m,m);
        for (int i = 0; i < m; i++) {
            S(i,i) = s.evaluate1(t.count_theta(A_,b_,x)(i));
        }
        return u*S*A_;
    }
};
int main() {
    building_block bb(4,3);
    bb.check_mat();
    std::cout << std::endl;
    bb.check_vec();
    std::cout << std::endl;
    Eigen::VectorXf x(3);
    x << 1,0,1;
    Eigen::RowVectorXf u(4);
    u << -1,2,0,1;
    std::cout << bb.count_grad_x(x,u);
    return 0;
}
