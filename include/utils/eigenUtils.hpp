//
// Created by xubo on 20-9-10.
//

#ifndef VANISHINGPOINT_EIGEN_UTILS_HPP
#define VANISHINGPOINT_EIGEN_UTILS_HPP

#include <eigen3/Eigen/Dense>

class Utility {
public:
    template<typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 3, 3> skewSymmetric(const Eigen::MatrixBase<Derived> &q) {
        Eigen::Matrix<typename Derived::Scalar, 3, 3> ans;
        ans << typename Derived::Scalar(0), -q(2), q(1),
                q(2), typename Derived::Scalar(0), -q(0),
                -q(1), q(0), typename Derived::Scalar(0);
        return ans;
    }

    template<typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 3, 3> ypr2R(const Eigen::MatrixBase<Derived> &ypr) {
        typedef typename Derived::Scalar Scalar_t;

        Scalar_t y = ypr(0) / 180.0 * M_PI;
        Scalar_t p = ypr(1) / 180.0 * M_PI;
        Scalar_t r = ypr(2) / 180.0 * M_PI;

        Eigen::Matrix<Scalar_t, 3, 3> Rz;
        Rz << cos(y), -sin(y), 0,
                sin(y), cos(y), 0,
                0, 0, 1;

        Eigen::Matrix<Scalar_t, 3, 3> Ry;
        Ry << cos(p), 0., sin(p),
                0., 1., 0.,
                -sin(p), 0., cos(p);

        Eigen::Matrix<Scalar_t, 3, 3> Rx;
        Rx << 1., 0., 0.,
                0., cos(r), -sin(r),
                0., sin(r), cos(r);

        return Rz * Ry * Rx;
    }

    static Eigen::Vector3d R2ypr(const Eigen::Matrix3d &R) {
        Eigen::Vector3d n = R.col(0);
        Eigen::Vector3d o = R.col(1);
        Eigen::Vector3d a = R.col(2);

        Eigen::Vector3d ypr(3);
        double y = atan2(n(1), n(0));
        double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
        double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
        ypr(0) = y;
        ypr(1) = p;
        ypr(2) = r;

        return ypr / M_PI * 180.0;
    }

    static Eigen::Matrix3d g2R(const Eigen::Vector3d &g) {
        Eigen::Matrix3d R0;
        Eigen::Vector3d ng1 = g.normalized();
        Eigen::Vector3d ng2{0, 0, 1.0};
        R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();
        double yaw = R2ypr(R0).x();
        R0 = ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
        // R0 = Utility::ypr2R(Eigen::Vector3d{-90, 0, 0}) * R0;
        return R0;
    }
};

#endif //VANISHINGPOINT_EIGEN_UTILS_HPP
