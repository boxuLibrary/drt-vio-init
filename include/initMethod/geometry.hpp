//
// Created by pi on 2020/2/11.
//

#ifndef GEOMETRY_HPP
#define GEOMETRY_HPP

#include <Eigen/Eigen>
#include<numeric>
#include <ceres/jet.h>

class Transformation {
public:
    Eigen::Vector3d _t;
    Eigen::Matrix3d _R;

public:
    Transformation() :
            _t(Eigen::Vector3d::Zero()),
            _R(Eigen::Matrix3d::Identity()) {}

    Transformation(const Eigen::Vector3d &t,
                   const Eigen::Matrix3d &R) :
            _t(t),
            _R(R) {}

    void SetIdentity() {
        _t.setZero();
        _R.setIdentity();
    }

    Transformation inverse() const {
        return Transformation(-_R.transpose() * _t, _R.transpose());
    }

    Transformation operator*(const Transformation &T) const {
        return Transformation(_R * T._t + _t, _R * T._R);
    }

    Eigen::Vector3d operator*(const Eigen::Vector3d &v) const {
        return _R * v + _t;
    }
};

static Eigen::Matrix3d Skew(const Eigen::Vector3d &v) {
    Eigen::Matrix3d v_x;
    v_x << 0, -v(2), v(1),
            v(2), 0, -v(0),
            -v(1), v(0), 0;
    return v_x;
}

static double SampsonError(const Eigen::Matrix3d &E,
                           const Eigen::Vector3d &f_0, const Eigen::Vector3d &f_1) {
    Eigen::Vector3d f_i = f_0 / f_0(2);
    Eigen::Vector3d f_j = f_1 / f_1(2);

    Eigen::Vector3d epi_line_i = E * f_j;
    Eigen::Vector3d epi_line_j = E.transpose() * f_i;

    double err = f_i.dot(epi_line_i);
    return std::fabs(err) / std::sqrt(epi_line_i.head(2).squaredNorm() + epi_line_j.head(2).squaredNorm());
}

static double SampsonError(const Eigen::Vector3d &t, const Eigen::Matrix3d &R,
                           const Eigen::Vector3d &f_0, const Eigen::Vector3d &f_1) {
    Eigen::Matrix3d E = Skew(t.normalized()) * R;

    Eigen::Vector3d f_i = f_0 / f_0(2);
    Eigen::Vector3d f_j = f_1 / f_1(2);

    Eigen::Vector3d epi_line_i = E * f_j;
    Eigen::Vector3d epi_line_j = E.transpose() * f_i;

    double err = f_i.dot(epi_line_i);
    return std::fabs(err) / std::sqrt(epi_line_i.head(2).squaredNorm() + epi_line_j.head(2).squaredNorm());
}

static bool CheckPointInFrontOfViews(
        const Eigen::Vector3d &f_0, const Eigen::Vector3d &f_1, const Transformation &T, double th = 50.0) {
    Eigen::Matrix<double, 3, 2> A;
    A << f_0, -T._R * f_1;
    Eigen::Vector2d lambda = (A.transpose() * A).inverse() * A.transpose() * T._t;
    return lambda(0) > 0 && lambda(1) > 0 && lambda(0) < th && lambda(1) < th;
}

static void DecomposeEssentialMatrix(
        const Eigen::Matrix3d &E, Eigen::Vector3d &t, Eigen::Matrix3d &R1, Eigen::Matrix3d &R2) {
    Eigen::Matrix3d W;
    W << 0, 1, 0, -1, 0, 0, 0, 0, 1;

    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    if (U.determinant() < 0)
        U.col(2) *= -1.0;
    if (V.determinant() < 0)
        V.col(2) *= -1.0;

    R1 = U * W * V.transpose();
    R2 = U * W.transpose() * V.transpose();
    t = U.col(2).normalized();
}

static int RecoverPoseFromEssentialMatrix(const Eigen::Matrix3d &E,
                                          const std::vector<Eigen::Vector3d> &f_0_vec,
                                          const std::vector<Eigen::Vector3d> &f_1_vec,
                                          std::vector<bool> &inlier_flags,
                                          Transformation &T) {
    Eigen::Vector3d t;
    Eigen::Matrix3d R1, R2;
    DecomposeEssentialMatrix(E, t, R1, R2);

    std::vector<Transformation> T_vec = {Transformation(t, R1), Transformation(-t, R1),
                                         Transformation(t, R2), Transformation(-t, R2)};

    std::vector<int> inlier_count_vec(T_vec.size(), 0);
    std::vector<std::vector<bool> > inlier_flags_vec(T_vec.size());
    for (size_t i = 0; i < T_vec.size(); ++i) {
        inlier_flags_vec[i] = std::vector<bool>(f_0_vec.size(), false);
        for (size_t j = 0; j < f_0_vec.size(); ++j) {
            if (!inlier_flags.empty() && !inlier_flags[j])
                continue;
            if (!CheckPointInFrontOfViews(f_0_vec[j], f_1_vec[j], T_vec[i]))
                continue;
            ++inlier_count_vec[i];
            inlier_flags_vec[i][j] = true;
        }
    }

    const auto &max_inlier_count_it = std::max_element(inlier_count_vec.begin(), inlier_count_vec.end());
    const int max_id = std::distance(inlier_count_vec.begin(), max_inlier_count_it);
    T = T_vec[max_id];

    if (!inlier_flags.empty()) {
        for (size_t i = 0; i < inlier_flags.size(); ++i)
            inlier_flags[i] = inlier_flags[i] && inlier_flags_vec[max_id][i];
        return std::accumulate(inlier_flags.begin(), inlier_flags.end(), 0);
    }
    inlier_flags = inlier_flags_vec[max_id];
    return *max_inlier_count_it;
}

static Eigen::Matrix3d Cayley2Rot(const Eigen::Vector3d &cayley) {
    Eigen::Matrix3d R;
    double scale = 1 + pow(cayley[0], 2) + pow(cayley[1], 2) + pow(cayley[2], 2);

    R(0, 0) = 1 + pow(cayley[0], 2) - pow(cayley[1], 2) - pow(cayley[2], 2);
    R(0, 1) = 2 * (cayley[0] * cayley[1] - cayley[2]);
    R(0, 2) = 2 * (cayley[0] * cayley[2] + cayley[1]);
    R(1, 0) = 2 * (cayley[0] * cayley[1] + cayley[2]);
    R(1, 1) = 1 - pow(cayley[0], 2) + pow(cayley[1], 2) - pow(cayley[2], 2);
    R(1, 2) = 2 * (cayley[1] * cayley[2] - cayley[0]);
    R(2, 0) = 2 * (cayley[0] * cayley[2] - cayley[1]);
    R(2, 1) = 2 * (cayley[1] * cayley[2] + cayley[0]);
    R(2, 2) = 1 - pow(cayley[0], 2) - pow(cayley[1], 2) + pow(cayley[2], 2);

    R = (1 / scale) * R;
    return R;
}

static Eigen::Vector3d Rot2Cayley(const Eigen::Matrix3d &R) {
    Eigen::Matrix3d C1;
    Eigen::Matrix3d C2;
    Eigen::Matrix3d C;
    C1 = R - Eigen::Matrix3d::Identity();
    C2 = R + Eigen::Matrix3d::Identity();
    C = C1 * C2.inverse();

    Eigen::Vector3d cayley;
    cayley[0] = -C(1, 2);
    cayley[1] = C(0, 2);
    cayley[2] = -C(0, 1);

    return cayley;
}


static Eigen::Matrix3d Cayley2RotReduced(const Eigen::Vector3d &cayley) {
    Eigen::Matrix3d R;

    R(0, 0) = 1 + pow(cayley[0], 2) - pow(cayley[1], 2) - pow(cayley[2], 2);
    R(0, 1) = 2 * (cayley[0] * cayley[1] - cayley[2]);
    R(0, 2) = 2 * (cayley[0] * cayley[2] + cayley[1]);
    R(1, 0) = 2 * (cayley[0] * cayley[1] + cayley[2]);
    R(1, 1) = 1 - pow(cayley[0], 2) + pow(cayley[1], 2) - pow(cayley[2], 2);
    R(1, 2) = 2 * (cayley[1] * cayley[2] - cayley[0]);
    R(2, 0) = 2 * (cayley[0] * cayley[2] - cayley[1]);
    R(2, 1) = 2 * (cayley[1] * cayley[2] + cayley[0]);
    R(2, 2) = 1 - pow(cayley[0], 2) - pow(cayley[1], 2) + pow(cayley[2], 2);

    return R;
}

static Eigen::Vector3d Quaternion2Cayley(const Eigen::Quaterniond& q)
{

    Eigen::Vector3d Cayley;

    Cayley.x() = q.x() / q.w();
    Cayley.y() = q.y() / q.w();
    Cayley.z() = q.z() / q.w();

    return Cayley;
}

template <typename T>
static Eigen::Matrix<T, 3, 1> Quaternion2Cayley(const Eigen::Quaternion<T>& q)
{

    Eigen::Matrix<T, 3, 1> Cayley;

    Cayley.x() = q.x() / q.w();
    Cayley.y() = q.y() / q.w();
    Cayley.z() = q.z() / q.w();

    return Cayley;
}

template <typename T>
static Eigen::Matrix<T, 3, 3> Cayley2RotReduced(const Eigen::Matrix<T, 3, 1> &cayley) {
    Eigen::Matrix<T, 3, 3> R;

    R(0, 0) = T(1) + ceres::pow(cayley[0], 2) - ceres::pow(cayley[1], 2) - ceres::pow(cayley[2], 2);
    R(0, 1) = T(2) * (cayley[0] * cayley[1] - cayley[2]);
    R(0, 2) = T(2) * (cayley[0] * cayley[2] + cayley[1]);
    R(1, 0) = T(2) * (cayley[0] * cayley[1] + cayley[2]);
    R(1, 1) = T(1) - ceres::pow(cayley[0], 2) + ceres::pow(cayley[1], 2) - ceres::pow(cayley[2], 2);
    R(1, 2) = T(2) * (cayley[1] * cayley[2] - cayley[0]);
    R(2, 0) = T(2) * (cayley[0] * cayley[2] - cayley[1]);
    R(2, 1) = T(2) * (cayley[1] * cayley[2] + cayley[0]);
    R(2, 2) = T(1) - ceres::pow(cayley[0], 2) - ceres::pow(cayley[1], 2) + ceres::pow(cayley[2], 2);

    return R;
}


#endif //GEOMETRY_HPP
