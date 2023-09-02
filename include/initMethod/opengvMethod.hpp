//
// Created by pi on 2020/3/7.
//

#ifndef OPENGV_EIGEN_SOLVER_HPP
#define OPENGV_EIGEN_SOLVER_HPP

#include <Eigen/Eigen>

#include "geometry.hpp"

using namespace std;

namespace opengv {

    static Eigen::Matrix3d ComposeMwithJacobians(
            const Eigen::Matrix3d &xxF,
            const Eigen::Matrix3d &yyF,
            const Eigen::Matrix3d &zzF,
            const Eigen::Matrix3d &xyF,
            const Eigen::Matrix3d &yzF,
            const Eigen::Matrix3d &zxF,
            const Eigen::Vector3d &cayley,
            Eigen::Matrix3d &M_jac1,
            Eigen::Matrix3d &M_jac2,
            Eigen::Matrix3d &M_jac3) {

        Eigen::Matrix3d R = Cayley2RotReduced(cayley);
        Eigen::Matrix3d R_jac1;
        Eigen::Matrix3d R_jac2;
        Eigen::Matrix3d R_jac3;

        R_jac1(0, 0) = 2 * cayley[0];
        R_jac1(0, 1) = 2 * cayley[1];
        R_jac1(0, 2) = 2 * cayley[2];
        R_jac1(1, 0) = 2 * cayley[1];
        R_jac1(1, 1) = -2 * cayley[0];
        R_jac1(1, 2) = -2;
        R_jac1(2, 0) = 2 * cayley[2];
        R_jac1(2, 1) = 2;
        R_jac1(2, 2) = -2 * cayley[0];

        R_jac2(0, 0) = -2 * cayley[1];
        R_jac2(0, 1) = 2 * cayley[0];
        R_jac2(0, 2) = 2;
        R_jac2(1, 0) = 2 * cayley[0];
        R_jac2(1, 1) = 2 * cayley[1];
        R_jac2(1, 2) = 2 * cayley[2];
        R_jac2(2, 0) = -2;
        R_jac2(2, 1) = 2 * cayley[2];
        R_jac2(2, 2) = -2 * cayley[1];

        R_jac3(0, 0) = -2 * cayley[2];
        R_jac3(0, 1) = -2;
        R_jac3(0, 2) = 2 * cayley[0];
        R_jac3(1, 0) = 2;
        R_jac3(1, 1) = -2 * cayley[2];
        R_jac3(1, 2) = 2 * cayley[1];
        R_jac3(2, 0) = 2 * cayley[0];
        R_jac3(2, 1) = 2 * cayley[1];
        R_jac3(2, 2) = 2 * cayley[2];

        //Fill the matrix M using the precomputed summation terms. Plus Jacobian.
        Eigen::Matrix3d M;
        double temp;
        temp = R.row(2) * yyF * R.row(2).transpose();
        M(0, 0) = temp;
        temp = -2.0 * R.row(2) * yzF * R.row(1).transpose();
        M(0, 0) += temp;
        temp = R.row(1) * zzF * R.row(1).transpose();
        M(0, 0) += temp;
        temp = 2.0 * R_jac1.row(2) * yyF * R.row(2).transpose();
        M_jac1(0, 0) = temp;
        temp = -2.0 * R_jac1.row(2) * yzF * R.row(1).transpose();
        M_jac1(0, 0) += temp;
        temp = -2.0 * R.row(2) * yzF * R_jac1.row(1).transpose();
        M_jac1(0, 0) += temp;
        temp = 2.0 * R_jac1.row(1) * zzF * R.row(1).transpose();
        M_jac1(0, 0) += temp;
        temp = 2.0 * R_jac2.row(2) * yyF * R.row(2).transpose();
        M_jac2(0, 0) = temp;
        temp = -2.0 * R_jac2.row(2) * yzF * R.row(1).transpose();
        M_jac2(0, 0) += temp;
        temp = -2.0 * R.row(2) * yzF * R_jac2.row(1).transpose();
        M_jac2(0, 0) += temp;
        temp = 2.0 * R_jac2.row(1) * zzF * R.row(1).transpose();
        M_jac2(0, 0) += temp;
        temp = 2.0 * R_jac3.row(2) * yyF * R.row(2).transpose();
        M_jac3(0, 0) = temp;
        temp = -2.0 * R_jac3.row(2) * yzF * R.row(1).transpose();
        M_jac3(0, 0) += temp;
        temp = -2.0 * R.row(2) * yzF * R_jac3.row(1).transpose();
        M_jac3(0, 0) += temp;
        temp = 2.0 * R_jac3.row(1) * zzF * R.row(1).transpose();
        M_jac3(0, 0) += temp;

        temp = R.row(2) * yzF * R.row(0).transpose();
        M(0, 1) = temp;
        temp = -1.0 * R.row(2) * xyF * R.row(2).transpose();
        M(0, 1) += temp;
        temp = -1.0 * R.row(1) * zzF * R.row(0).transpose();
        M(0, 1) += temp;
        temp = R.row(1) * zxF * R.row(2).transpose();
        M(0, 1) += temp;
        temp = R_jac1.row(2) * yzF * R.row(0).transpose();
        M_jac1(0, 1) = temp;
        temp = R.row(2) * yzF * R_jac1.row(0).transpose();
        M_jac1(0, 1) += temp;
        temp = -2.0 * R_jac1.row(2) * xyF * R.row(2).transpose();
        M_jac1(0, 1) += temp;
        temp = -R_jac1.row(1) * zzF * R.row(0).transpose();
        M_jac1(0, 1) += temp;
        temp = -R.row(1) * zzF * R_jac1.row(0).transpose();
        M_jac1(0, 1) += temp;
        temp = R_jac1.row(1) * zxF * R.row(2).transpose();
        M_jac1(0, 1) += temp;
        temp = R.row(1) * zxF * R_jac1.row(2).transpose();
        M_jac1(0, 1) += temp;
        temp = R_jac2.row(2) * yzF * R.row(0).transpose();
        M_jac2(0, 1) = temp;
        temp = R.row(2) * yzF * R_jac2.row(0).transpose();
        M_jac2(0, 1) += temp;
        temp = -2.0 * R_jac2.row(2) * xyF * R.row(2).transpose();
        M_jac2(0, 1) += temp;
        temp = -R_jac2.row(1) * zzF * R.row(0).transpose();
        M_jac2(0, 1) += temp;
        temp = -R.row(1) * zzF * R_jac2.row(0).transpose();
        M_jac2(0, 1) += temp;
        temp = R_jac2.row(1) * zxF * R.row(2).transpose();
        M_jac2(0, 1) += temp;
        temp = R.row(1) * zxF * R_jac2.row(2).transpose();
        M_jac2(0, 1) += temp;
        temp = R_jac3.row(2) * yzF * R.row(0).transpose();
        M_jac3(0, 1) = temp;
        temp = R.row(2) * yzF * R_jac3.row(0).transpose();
        M_jac3(0, 1) += temp;
        temp = -2.0 * R_jac3.row(2) * xyF * R.row(2).transpose();
        M_jac3(0, 1) += temp;
        temp = -R_jac3.row(1) * zzF * R.row(0).transpose();
        M_jac3(0, 1) += temp;
        temp = -R.row(1) * zzF * R_jac3.row(0).transpose();
        M_jac3(0, 1) += temp;
        temp = R_jac3.row(1) * zxF * R.row(2).transpose();
        M_jac3(0, 1) += temp;
        temp = R.row(1) * zxF * R_jac3.row(2).transpose();
        M_jac3(0, 1) += temp;

        temp = R.row(2) * xyF * R.row(1).transpose();
        M(0, 2) = temp;
        temp = -1.0 * R.row(2) * yyF * R.row(0).transpose();
        M(0, 2) += temp;
        temp = -1.0 * R.row(1) * zxF * R.row(1).transpose();
        M(0, 2) += temp;
        temp = R.row(1) * yzF * R.row(0).transpose();
        M(0, 2) += temp;
        temp = R_jac1.row(2) * xyF * R.row(1).transpose();
        M_jac1(0, 2) = temp;
        temp = R.row(2) * xyF * R_jac1.row(1).transpose();
        M_jac1(0, 2) += temp;
        temp = -R_jac1.row(2) * yyF * R.row(0).transpose();
        M_jac1(0, 2) += temp;
        temp = -R.row(2) * yyF * R_jac1.row(0).transpose();
        M_jac1(0, 2) += temp;
        temp = -2.0 * R_jac1.row(1) * zxF * R.row(1).transpose();
        M_jac1(0, 2) += temp;
        temp = R_jac1.row(1) * yzF * R.row(0).transpose();
        M_jac1(0, 2) += temp;
        temp = R.row(1) * yzF * R_jac1.row(0).transpose();
        M_jac1(0, 2) += temp;
        temp = R_jac2.row(2) * xyF * R.row(1).transpose();
        M_jac2(0, 2) = temp;
        temp = R.row(2) * xyF * R_jac2.row(1).transpose();
        M_jac2(0, 2) += temp;
        temp = -R_jac2.row(2) * yyF * R.row(0).transpose();
        M_jac2(0, 2) += temp;
        temp = -R.row(2) * yyF * R_jac2.row(0).transpose();
        M_jac2(0, 2) += temp;
        temp = -2.0 * R_jac2.row(1) * zxF * R.row(1).transpose();
        M_jac2(0, 2) += temp;
        temp = R_jac2.row(1) * yzF * R.row(0).transpose();
        M_jac2(0, 2) += temp;
        temp = R.row(1) * yzF * R_jac2.row(0).transpose();
        M_jac2(0, 2) += temp;
        temp = R_jac3.row(2) * xyF * R.row(1).transpose();
        M_jac3(0, 2) = temp;
        temp = R.row(2) * xyF * R_jac3.row(1).transpose();
        M_jac3(0, 2) += temp;
        temp = -R_jac3.row(2) * yyF * R.row(0).transpose();
        M_jac3(0, 2) += temp;
        temp = -R.row(2) * yyF * R_jac3.row(0).transpose();
        M_jac3(0, 2) += temp;
        temp = -2.0 * R_jac3.row(1) * zxF * R.row(1).transpose();
        M_jac3(0, 2) += temp;
        temp = R_jac3.row(1) * yzF * R.row(0).transpose();
        M_jac3(0, 2) += temp;
        temp = R.row(1) * yzF * R_jac3.row(0).transpose();
        M_jac3(0, 2) += temp;

        temp = R.row(0) * zzF * R.row(0).transpose();
        M(1, 1) = temp;
        temp = -2.0 * R.row(0) * zxF * R.row(2).transpose();
        M(1, 1) += temp;
        temp = R.row(2) * xxF * R.row(2).transpose();
        M(1, 1) += temp;
        temp = 2.0 * R_jac1.row(0) * zzF * R.row(0).transpose();
        M_jac1(1, 1) = temp;
        temp = -2.0 * R_jac1.row(0) * zxF * R.row(2).transpose();
        M_jac1(1, 1) += temp;
        temp = -2.0 * R.row(0) * zxF * R_jac1.row(2).transpose();
        M_jac1(1, 1) += temp;
        temp = 2.0 * R_jac1.row(2) * xxF * R.row(2).transpose();
        M_jac1(1, 1) += temp;
        temp = 2.0 * R_jac2.row(0) * zzF * R.row(0).transpose();
        M_jac2(1, 1) = temp;
        temp = -2.0 * R_jac2.row(0) * zxF * R.row(2).transpose();
        M_jac2(1, 1) += temp;
        temp = -2.0 * R.row(0) * zxF * R_jac2.row(2).transpose();
        M_jac2(1, 1) += temp;
        temp = 2.0 * R_jac2.row(2) * xxF * R.row(2).transpose();
        M_jac2(1, 1) += temp;
        temp = 2.0 * R_jac3.row(0) * zzF * R.row(0).transpose();
        M_jac3(1, 1) = temp;
        temp = -2.0 * R_jac3.row(0) * zxF * R.row(2).transpose();
        M_jac3(1, 1) += temp;
        temp = -2.0 * R.row(0) * zxF * R_jac3.row(2).transpose();
        M_jac3(1, 1) += temp;
        temp = 2.0 * R_jac3.row(2) * xxF * R.row(2).transpose();
        M_jac3(1, 1) += temp;

        temp = R.row(0) * zxF * R.row(1).transpose();
        M(1, 2) = temp;
        temp = -1.0 * R.row(0) * yzF * R.row(0).transpose();
        M(1, 2) += temp;
        temp = -1.0 * R.row(2) * xxF * R.row(1).transpose();
        M(1, 2) += temp;
        temp = R.row(2) * xyF * R.row(0).transpose();
        M(1, 2) += temp;
        temp = R_jac1.row(0) * zxF * R.row(1).transpose();
        M_jac1(1, 2) = temp;
        temp = R.row(0) * zxF * R_jac1.row(1).transpose();
        M_jac1(1, 2) += temp;
        temp = -2.0 * R_jac1.row(0) * yzF * R.row(0).transpose();
        M_jac1(1, 2) += temp;
        temp = -R_jac1.row(2) * xxF * R.row(1).transpose();
        M_jac1(1, 2) += temp;
        temp = -R.row(2) * xxF * R_jac1.row(1).transpose();
        M_jac1(1, 2) += temp;
        temp = R_jac1.row(2) * xyF * R.row(0).transpose();
        M_jac1(1, 2) += temp;
        temp = R.row(2) * xyF * R_jac1.row(0).transpose();
        M_jac1(1, 2) += temp;
        temp = R_jac2.row(0) * zxF * R.row(1).transpose();
        M_jac2(1, 2) = temp;
        temp = R.row(0) * zxF * R_jac2.row(1).transpose();
        M_jac2(1, 2) += temp;
        temp = -2.0 * R_jac2.row(0) * yzF * R.row(0).transpose();
        M_jac2(1, 2) += temp;
        temp = -R_jac2.row(2) * xxF * R.row(1).transpose();
        M_jac2(1, 2) += temp;
        temp = -R.row(2) * xxF * R_jac2.row(1).transpose();
        M_jac2(1, 2) += temp;
        temp = R_jac2.row(2) * xyF * R.row(0).transpose();
        M_jac2(1, 2) += temp;
        temp = R.row(2) * xyF * R_jac2.row(0).transpose();
        M_jac2(1, 2) += temp;
        temp = R_jac3.row(0) * zxF * R.row(1).transpose();
        M_jac3(1, 2) = temp;
        temp = R.row(0) * zxF * R_jac3.row(1).transpose();
        M_jac3(1, 2) += temp;
        temp = -2.0 * R_jac3.row(0) * yzF * R.row(0).transpose();
        M_jac3(1, 2) += temp;
        temp = -R_jac3.row(2) * xxF * R.row(1).transpose();
        M_jac3(1, 2) += temp;
        temp = -R.row(2) * xxF * R_jac3.row(1).transpose();
        M_jac3(1, 2) += temp;
        temp = R_jac3.row(2) * xyF * R.row(0).transpose();
        M_jac3(1, 2) += temp;
        temp = R.row(2) * xyF * R_jac3.row(0).transpose();
        M_jac3(1, 2) += temp;

        temp = R.row(1) * xxF * R.row(1).transpose();
        M(2, 2) = temp;
        temp = -2.0 * R.row(0) * xyF * R.row(1).transpose();
        M(2, 2) += temp;
        temp = R.row(0) * yyF * R.row(0).transpose();
        M(2, 2) += temp;
        temp = 2.0 * R_jac1.row(1) * xxF * R.row(1).transpose();
        M_jac1(2, 2) = temp;
        temp = -2.0 * R_jac1.row(0) * xyF * R.row(1).transpose();
        M_jac1(2, 2) += temp;
        temp = -2.0 * R.row(0) * xyF * R_jac1.row(1).transpose();
        M_jac1(2, 2) += temp;
        temp = 2.0 * R_jac1.row(0) * yyF * R.row(0).transpose();
        M_jac1(2, 2) += temp;
        temp = 2.0 * R_jac2.row(1) * xxF * R.row(1).transpose();
        M_jac2(2, 2) = temp;
        temp = -2.0 * R_jac2.row(0) * xyF * R.row(1).transpose();
        M_jac2(2, 2) += temp;
        temp = -2.0 * R.row(0) * xyF * R_jac2.row(1).transpose();
        M_jac2(2, 2) += temp;
        temp = 2.0 * R_jac2.row(0) * yyF * R.row(0).transpose();
        M_jac2(2, 2) += temp;
        temp = 2.0 * R_jac3.row(1) * xxF * R.row(1).transpose();
        M_jac3(2, 2) = temp;
        temp = -2.0 * R_jac3.row(0) * xyF * R.row(1).transpose();
        M_jac3(2, 2) += temp;
        temp = -2.0 * R.row(0) * xyF * R_jac3.row(1).transpose();
        M_jac3(2, 2) += temp;
        temp = 2.0 * R_jac3.row(0) * yyF * R.row(0).transpose();
        M_jac3(2, 2) += temp;

        M(1, 0) = M(0, 1);
        M(2, 0) = M(0, 2);
        M(2, 1) = M(1, 2);
        M_jac1(1, 0) = M_jac1(0, 1);
        M_jac1(2, 0) = M_jac1(0, 2);
        M_jac1(2, 1) = M_jac1(1, 2);
        M_jac2(1, 0) = M_jac2(0, 1);
        M_jac2(2, 0) = M_jac2(0, 2);
        M_jac2(2, 1) = M_jac2(1, 2);
        M_jac3(1, 0) = M_jac3(0, 1);
        M_jac3(2, 0) = M_jac3(0, 2);
        M_jac3(2, 1) = M_jac3(1, 2);

        return M;
    }



    template <typename T>
    static Eigen::Matrix<T, 3, 3> ComposeMwithJacobians(
            const Eigen::Matrix3d &xxF,
            const Eigen::Matrix3d &yyF,
            const Eigen::Matrix3d &zzF,
            const Eigen::Matrix3d &xyF,
            const Eigen::Matrix3d &yzF,
            const Eigen::Matrix3d &zxF,
            const Eigen::Matrix<T, 3, 1> &cayley,
            Eigen::Matrix<T, 3, 3> &M_jac1,
            Eigen::Matrix<T, 3, 3> &M_jac2,
            Eigen::Matrix<T, 3, 3> &M_jac3) {

        using Matrix3T = Eigen::Matrix<T, 3, 3>;



        Matrix3T R = Cayley2RotReduced(cayley);
        Matrix3T R_jac1;
        Matrix3T R_jac2;
        Matrix3T R_jac3;

        R_jac1(0, 0) =  T(2) * cayley[0];
        R_jac1(0, 1) =  T(2) * cayley[1];
        R_jac1(0, 2) =  T(2) * cayley[2];
        R_jac1(1, 0) =  T(2) * cayley[1];
        R_jac1(1, 1) = -T(2) * cayley[0];
        R_jac1(1, 2) = -T(2);
        R_jac1(2, 0) =  T(2) * cayley[2];
        R_jac1(2, 1) =  T(2);
        R_jac1(2, 2) = -T(2) * cayley[0];
        R_jac2(0, 0) = -T(2) * cayley[1];
        R_jac2(0, 1) =  T(2) * cayley[0];
        R_jac2(0, 2) =  T(2);
        R_jac2(1, 0) =  T(2) * cayley[0];
        R_jac2(1, 1) =  T(2) * cayley[1];
        R_jac2(1, 2) =  T(2) * cayley[2];
        R_jac2(2, 0) = -T(2);
        R_jac2(2, 1) =  T(2) * cayley[2];
        R_jac2(2, 2) = -T(2) * cayley[1];
        R_jac3(0, 0) = -T(2) * cayley[2];
        R_jac3(0, 1) = -T(2);
        R_jac3(0, 2) =  T(2) * cayley[0];
        R_jac3(1, 0) =  T(2);
        R_jac3(1, 1) = -T(2) * cayley[2];
        R_jac3(1, 2) =  T(2) * cayley[1];
        R_jac3(2, 0) =  T(2) * cayley[0];
        R_jac3(2, 1) =  T(2) * cayley[1];
        R_jac3(2, 2) =  T(2) * cayley[2];

        //Fill the matrix M using the precomputed summation terms. Plus Jacobian.
        Matrix3T M;
        Eigen::Matrix<T, 1, 1> temp;
        temp = R.row(2) * yyF.cast<T>() * R.row(2).transpose();
        M(0, 0) = temp(0, 0);
        temp = -2.0 * R.row(2) * yzF.cast<T>() * R.row(1).transpose();
        M(0, 0) += temp(0, 0);
        temp = R.row(1) * zzF.cast<T>() * R.row(1).transpose();
        M(0, 0) += temp(0, 0);
        temp = 2.0 * R_jac1.row(2) * yyF.cast<T>() * R.row(2).transpose();
        M_jac1(0, 0) = temp(0, 0);
        temp = -2.0 * R_jac1.row(2) * yzF.cast<T>() * R.row(1).transpose();
        M_jac1(0, 0) += temp(0, 0);
        temp = -2.0 * R.row(2) * yzF.cast<T>() * R_jac1.row(1).transpose();
        M_jac1(0, 0) += temp(0, 0);
        temp = 2.0 * R_jac1.row(1) * zzF.cast<T>() * R.row(1).transpose();
        M_jac1(0, 0) += temp(0, 0);
        temp = 2.0 * R_jac2.row(2) * yyF.cast<T>() * R.row(2).transpose();
        M_jac2(0, 0) = temp(0, 0);
        temp = -2.0 * R_jac2.row(2) * yzF.cast<T>() * R.row(1).transpose();
        M_jac2(0, 0) += temp(0, 0);
        temp = -2.0 * R.row(2) * yzF.cast<T>() * R_jac2.row(1).transpose();
        M_jac2(0, 0) += temp(0, 0);
        temp = 2.0 * R_jac2.row(1) * zzF.cast<T>() * R.row(1).transpose();
        M_jac2(0, 0) += temp(0, 0);
        temp = 2.0 * R_jac3.row(2) * yyF.cast<T>() * R.row(2).transpose();
        M_jac3(0, 0) = temp(0, 0);
        temp = -2.0 * R_jac3.row(2) * yzF.cast<T>() * R.row(1).transpose();
        M_jac3(0, 0) += temp(0, 0);
        temp = -2.0 * R.row(2) * yzF.cast<T>() * R_jac3.row(1).transpose();
        M_jac3(0, 0) += temp(0, 0);
        temp = 2.0 * R_jac3.row(1) * zzF.cast<T>() * R.row(1).transpose();
        M_jac3(0, 0) += temp(0, 0);

        temp = R.row(2) * yzF.cast<T>() * R.row(0).transpose();
        M(0, 1) = temp(0, 0);
        temp = -1.0 * R.row(2) * xyF.cast<T>() * R.row(2).transpose();
        M(0, 1) += temp(0, 0);
        temp = -1.0 * R.row(1) * zzF.cast<T>() * R.row(0).transpose();
        M(0, 1) += temp(0, 0);
        temp = R.row(1) * zxF.cast<T>() * R.row(2).transpose();
        M(0, 1) += temp(0, 0);
        temp = R_jac1.row(2) * yzF.cast<T>() * R.row(0).transpose();
        M_jac1(0, 1) = temp(0, 0);
        temp = R.row(2) * yzF.cast<T>() * R_jac1.row(0).transpose();
        M_jac1(0, 1) += temp(0, 0);
        temp = -2.0 * R_jac1.row(2) * xyF.cast<T>() * R.row(2).transpose();
        M_jac1(0, 1) += temp(0, 0);
        temp = -R_jac1.row(1) * zzF.cast<T>() * R.row(0).transpose();
        M_jac1(0, 1) += temp(0, 0);
        temp = -R.row(1) * zzF.cast<T>() * R_jac1.row(0).transpose();
        M_jac1(0, 1) += temp(0, 0);
        temp = R_jac1.row(1) * zxF.cast<T>() * R.row(2).transpose();
        M_jac1(0, 1) += temp(0, 0);
        temp = R.row(1) * zxF.cast<T>() * R_jac1.row(2).transpose();
        M_jac1(0, 1) += temp(0, 0);
        temp = R_jac2.row(2) * yzF.cast<T>() * R.row(0).transpose();
        M_jac2(0, 1) = temp(0, 0);
        temp = R.row(2) * yzF.cast<T>() * R_jac2.row(0).transpose();
        M_jac2(0, 1) += temp(0, 0);
        temp = -2.0 * R_jac2.row(2) * xyF.cast<T>() * R.row(2).transpose();
        M_jac2(0, 1) += temp(0, 0);
        temp = -R_jac2.row(1) * zzF.cast<T>() * R.row(0).transpose();
        M_jac2(0, 1) += temp(0, 0);
        temp = -R.row(1) * zzF.cast<T>() * R_jac2.row(0).transpose();
        M_jac2(0, 1) += temp(0, 0);
        temp = R_jac2.row(1) * zxF.cast<T>() * R.row(2).transpose();
        M_jac2(0, 1) += temp(0, 0);
        temp = R.row(1) * zxF.cast<T>() * R_jac2.row(2).transpose();
        M_jac2(0, 1) += temp(0, 0);
        temp = R_jac3.row(2) * yzF.cast<T>() * R.row(0).transpose();
        M_jac3(0, 1) = temp(0, 0);
        temp = R.row(2) * yzF.cast<T>() * R_jac3.row(0).transpose();
        M_jac3(0, 1) += temp(0, 0);
        temp = -2.0 * R_jac3.row(2) * xyF.cast<T>() * R.row(2).transpose();
        M_jac3(0, 1) += temp(0, 0);
        temp = -R_jac3.row(1) * zzF.cast<T>() * R.row(0).transpose();
        M_jac3(0, 1) += temp(0, 0);
        temp = -R.row(1) * zzF.cast<T>() * R_jac3.row(0).transpose();
        M_jac3(0, 1) += temp(0, 0);
        temp = R_jac3.row(1) * zxF.cast<T>() * R.row(2).transpose();
        M_jac3(0, 1) += temp(0, 0);
        temp = R.row(1) * zxF.cast<T>() * R_jac3.row(2).transpose();
        M_jac3(0, 1) += temp(0, 0);

        temp = R.row(2) * xyF.cast<T>() * R.row(1).transpose();
        M(0, 2) = temp(0, 0);
        temp = -1.0 * R.row(2) * yyF.cast<T>() * R.row(0).transpose();
        M(0, 2) += temp(0, 0);
        temp = -1.0 * R.row(1) * zxF.cast<T>() * R.row(1).transpose();
        M(0, 2) += temp(0, 0);
        temp = R.row(1) * yzF.cast<T>() * R.row(0).transpose();
        M(0, 2) += temp(0, 0);
        temp = R_jac1.row(2) * xyF.cast<T>() * R.row(1).transpose();
        M_jac1(0, 2) = temp(0, 0);
        temp = R.row(2) * xyF.cast<T>() * R_jac1.row(1).transpose();
        M_jac1(0, 2) += temp(0, 0);
        temp = -R_jac1.row(2) * yyF.cast<T>() * R.row(0).transpose();
        M_jac1(0, 2) += temp(0, 0);
        temp = -R.row(2) * yyF.cast<T>() * R_jac1.row(0).transpose();
        M_jac1(0, 2) += temp(0, 0);
        temp = -2.0 * R_jac1.row(1) * zxF.cast<T>() * R.row(1).transpose();
        M_jac1(0, 2) += temp(0, 0);
        temp = R_jac1.row(1) * yzF.cast<T>() * R.row(0).transpose();
        M_jac1(0, 2) += temp(0, 0);
        temp = R.row(1) * yzF.cast<T>() * R_jac1.row(0).transpose();
        M_jac1(0, 2) += temp(0, 0);
        temp = R_jac2.row(2) * xyF.cast<T>() * R.row(1).transpose();
        M_jac2(0, 2) = temp(0, 0);
        temp = R.row(2) * xyF.cast<T>() * R_jac2.row(1).transpose();
        M_jac2(0, 2) += temp(0, 0);
        temp = -R_jac2.row(2) * yyF.cast<T>() * R.row(0).transpose();
        M_jac2(0, 2) += temp(0, 0);
        temp = -R.row(2) * yyF.cast<T>() * R_jac2.row(0).transpose();
        M_jac2(0, 2) += temp(0, 0);
        temp = -2.0 * R_jac2.row(1) * zxF.cast<T>() * R.row(1).transpose();
        M_jac2(0, 2) += temp(0, 0);
        temp = R_jac2.row(1) * yzF.cast<T>() * R.row(0).transpose();
        M_jac2(0, 2) += temp(0, 0);
        temp = R.row(1) * yzF.cast<T>() * R_jac2.row(0).transpose();
        M_jac2(0, 2) += temp(0, 0);
        temp = R_jac3.row(2) * xyF.cast<T>() * R.row(1).transpose();
        M_jac3(0, 2) = temp(0, 0);
        temp = R.row(2) * xyF.cast<T>() * R_jac3.row(1).transpose();
        M_jac3(0, 2) += temp(0, 0);
        temp = -R_jac3.row(2) * yyF.cast<T>() * R.row(0).transpose();
        M_jac3(0, 2) += temp(0, 0);
        temp = -R.row(2) * yyF.cast<T>() * R_jac3.row(0).transpose();
        M_jac3(0, 2) += temp(0, 0);
        temp = -2.0 * R_jac3.row(1) * zxF.cast<T>() * R.row(1).transpose();
        M_jac3(0, 2) += temp(0, 0);
        temp = R_jac3.row(1) * yzF.cast<T>() * R.row(0).transpose();
        M_jac3(0, 2) += temp(0, 0);
        temp = R.row(1) * yzF.cast<T>() * R_jac3.row(0).transpose();
        M_jac3(0, 2) += temp(0, 0);

        temp = R.row(0) * zzF.cast<T>() * R.row(0).transpose();
        M(1, 1) = temp(0, 0);
        temp = -2.0 * R.row(0) * zxF.cast<T>() * R.row(2).transpose();
        M(1, 1) += temp(0, 0);
        temp = R.row(2) * xxF.cast<T>() * R.row(2).transpose();
        M(1, 1) += temp(0, 0);
        temp = 2.0 * R_jac1.row(0) * zzF.cast<T>() * R.row(0).transpose();
        M_jac1(1, 1) = temp(0, 0);
        temp = -2.0 * R_jac1.row(0) * zxF.cast<T>() * R.row(2).transpose();
        M_jac1(1, 1) += temp(0, 0);
        temp = -2.0 * R.row(0) * zxF.cast<T>() * R_jac1.row(2).transpose();
        M_jac1(1, 1) += temp(0, 0);
        temp = 2.0 * R_jac1.row(2) * xxF.cast<T>() * R.row(2).transpose();
        M_jac1(1, 1) += temp(0, 0);
        temp = 2.0 * R_jac2.row(0) * zzF.cast<T>() * R.row(0).transpose();
        M_jac2(1, 1) = temp(0, 0);
        temp = -2.0 * R_jac2.row(0) * zxF.cast<T>() * R.row(2).transpose();
        M_jac2(1, 1) += temp(0, 0);
        temp = -2.0 * R.row(0) * zxF.cast<T>() * R_jac2.row(2).transpose();
        M_jac2(1, 1) += temp(0, 0);
        temp = 2.0 * R_jac2.row(2) * xxF.cast<T>() * R.row(2).transpose();
        M_jac2(1, 1) += temp(0, 0);
        temp = 2.0 * R_jac3.row(0) * zzF.cast<T>() * R.row(0).transpose();
        M_jac3(1, 1) = temp(0, 0);
        temp = -2.0 * R_jac3.row(0) * zxF.cast<T>() * R.row(2).transpose();
        M_jac3(1, 1) += temp(0, 0);
        temp = -2.0 * R.row(0) * zxF.cast<T>() * R_jac3.row(2).transpose();
        M_jac3(1, 1) += temp(0, 0);
        temp = 2.0 * R_jac3.row(2) * xxF.cast<T>() * R.row(2).transpose();
        M_jac3(1, 1) += temp(0, 0);

        temp = R.row(0) * zxF.cast<T>() * R.row(1).transpose();
        M(1, 2) = temp(0, 0);
        temp = -1.0 * R.row(0) * yzF.cast<T>() * R.row(0).transpose();
        M(1, 2) += temp(0, 0);
        temp = -1.0 * R.row(2) * xxF.cast<T>() * R.row(1).transpose();
        M(1, 2) += temp(0, 0);
        temp = R.row(2) * xyF.cast<T>() * R.row(0).transpose();
        M(1, 2) += temp(0, 0);
        temp = R_jac1.row(0) * zxF.cast<T>() * R.row(1).transpose();
        M_jac1(1, 2) = temp(0, 0);
        temp = R.row(0) * zxF.cast<T>() * R_jac1.row(1).transpose();
        M_jac1(1, 2) += temp(0, 0);
        temp = -2.0 * R_jac1.row(0) * yzF.cast<T>() * R.row(0).transpose();
        M_jac1(1, 2) += temp(0, 0);
        temp = -R_jac1.row(2) * xxF.cast<T>() * R.row(1).transpose();
        M_jac1(1, 2) += temp(0, 0);
        temp = -R.row(2) * xxF.cast<T>() * R_jac1.row(1).transpose();
        M_jac1(1, 2) += temp(0, 0);
        temp = R_jac1.row(2) * xyF.cast<T>() * R.row(0).transpose();
        M_jac1(1, 2) += temp(0, 0);
        temp = R.row(2) * xyF.cast<T>() * R_jac1.row(0).transpose();
        M_jac1(1, 2) += temp(0, 0);
        temp = R_jac2.row(0) * zxF.cast<T>() * R.row(1).transpose();
        M_jac2(1, 2) = temp(0, 0);
        temp = R.row(0) * zxF.cast<T>() * R_jac2.row(1).transpose();
        M_jac2(1, 2) += temp(0, 0);
        temp = -2.0 * R_jac2.row(0) * yzF.cast<T>() * R.row(0).transpose();
        M_jac2(1, 2) += temp(0, 0);
        temp = -R_jac2.row(2) * xxF.cast<T>() * R.row(1).transpose();
        M_jac2(1, 2) += temp(0, 0);
        temp = -R.row(2) * xxF.cast<T>() * R_jac2.row(1).transpose();
        M_jac2(1, 2) += temp(0, 0);
        temp = R_jac2.row(2) * xyF.cast<T>() * R.row(0).transpose();
        M_jac2(1, 2) += temp(0, 0);
        temp = R.row(2) * xyF.cast<T>() * R_jac2.row(0).transpose();
        M_jac2(1, 2) += temp(0, 0);
        temp = R_jac3.row(0) * zxF.cast<T>() * R.row(1).transpose();
        M_jac3(1, 2) = temp(0, 0);
        temp = R.row(0) * zxF.cast<T>() * R_jac3.row(1).transpose();
        M_jac3(1, 2) += temp(0, 0);
        temp = -2.0 * R_jac3.row(0) * yzF.cast<T>() * R.row(0).transpose();
        M_jac3(1, 2) += temp(0, 0);
        temp = -R_jac3.row(2) * xxF.cast<T>() * R.row(1).transpose();
        M_jac3(1, 2) += temp(0, 0);
        temp = -R.row(2) * xxF.cast<T>() * R_jac3.row(1).transpose();
        M_jac3(1, 2) += temp(0, 0);
        temp = R_jac3.row(2) * xyF.cast<T>() * R.row(0).transpose();
        M_jac3(1, 2) += temp(0, 0);
        temp = R.row(2) * xyF.cast<T>() * R_jac3.row(0).transpose();
        M_jac3(1, 2) += temp(0, 0);

        temp = R.row(1) * xxF.cast<T>() * R.row(1).transpose();
        M(2, 2) = temp(0, 0);
        temp = -2.0 * R.row(0) * xyF.cast<T>() * R.row(1).transpose();
        M(2, 2) += temp(0, 0);
        temp = R.row(0) * yyF.cast<T>() * R.row(0).transpose();
        M(2, 2) += temp(0, 0);
        temp = 2.0 * R_jac1.row(1) * xxF.cast<T>() * R.row(1).transpose();
        M_jac1(2, 2) = temp(0, 0);
        temp = -2.0 * R_jac1.row(0) * xyF.cast<T>() * R.row(1).transpose();
        M_jac1(2, 2) += temp(0, 0);
        temp = -2.0 * R.row(0) * xyF.cast<T>() * R_jac1.row(1).transpose();
        M_jac1(2, 2) += temp(0, 0);
        temp = 2.0 * R_jac1.row(0) * yyF.cast<T>() * R.row(0).transpose();
        M_jac1(2, 2) += temp(0, 0);
        temp = 2.0 * R_jac2.row(1) * xxF.cast<T>() * R.row(1).transpose();
        M_jac2(2, 2) = temp(0, 0);
        temp = -2.0 * R_jac2.row(0) * xyF.cast<T>() * R.row(1).transpose();
        M_jac2(2, 2) += temp(0, 0);
        temp = -2.0 * R.row(0) * xyF.cast<T>() * R_jac2.row(1).transpose();
        M_jac2(2, 2) += temp(0, 0);
        temp = 2.0 * R_jac2.row(0) * yyF.cast<T>() * R.row(0).transpose();
        M_jac2(2, 2) += temp(0, 0);
        temp = 2.0 * R_jac3.row(1) * xxF.cast<T>() * R.row(1).transpose();
        M_jac3(2, 2) = temp(0, 0);
        temp = -2.0 * R_jac3.row(0) * xyF.cast<T>() * R.row(1).transpose();
        M_jac3(2, 2) += temp(0, 0);
        temp = -2.0 * R.row(0) * xyF.cast<T>() * R_jac3.row(1).transpose();
        M_jac3(2, 2) += temp(0, 0);
        temp = 2.0 * R_jac3.row(0) * yyF.cast<T>() * R.row(0).transpose();
        M_jac3(2, 2) += temp(0, 0);

        M(1, 0) = M(0, 1);
        M(2, 0) = M(0, 2);
        M(2, 1) = M(1, 2);
        M_jac1(1, 0) = M_jac1(0, 1);
        M_jac1(2, 0) = M_jac1(0, 2);
        M_jac1(2, 1) = M_jac1(1, 2);
        M_jac2(1, 0) = M_jac2(0, 1);
        M_jac2(2, 0) = M_jac2(0, 2);
        M_jac2(2, 1) = M_jac2(1, 2);
        M_jac3(1, 0) = M_jac3(0, 1);
        M_jac3(2, 0) = M_jac3(0, 2);
        M_jac3(2, 1) = M_jac3(1, 2);

        return M;
    }

    static double GetSmallestEVwithJacobian(
            const Eigen::Matrix3d &xxF,
            const Eigen::Matrix3d &yyF,
            const Eigen::Matrix3d &zzF,
            const Eigen::Matrix3d &xyF,
            const Eigen::Matrix3d &yzF,
            const Eigen::Matrix3d &zxF,
            const Eigen::Vector3d &cayley,
            Eigen::Matrix<double, 1, 3> &jacobian) {
        Eigen::Matrix3d M_jac1 = Eigen::Matrix3d::Zero();
        Eigen::Matrix3d M_jac2 = Eigen::Matrix3d::Zero();
        Eigen::Matrix3d M_jac3 = Eigen::Matrix3d::Zero();

        Eigen::Matrix3d M = ComposeMwithJacobians(
                xxF, yyF, zzF, xyF, yzF, zxF, cayley, M_jac1, M_jac2, M_jac3);

        //Retrieve the smallest Eigenvalue by the following closed form solution.
        //Plus Jacobian.
        double b = -M(0, 0) - M(1, 1) - M(2, 2);
        double b_jac1 = -M_jac1(0, 0) - M_jac1(1, 1) - M_jac1(2, 2);
        double b_jac2 = -M_jac2(0, 0) - M_jac2(1, 1) - M_jac2(2, 2);
        double b_jac3 = -M_jac3(0, 0) - M_jac3(1, 1) - M_jac3(2, 2);
        double c =
                -pow(M(0, 2), 2) - pow(M(1, 2), 2) - pow(M(0, 1), 2) +
                M(0, 0) * M(1, 1) + M(0, 0) * M(2, 2) + M(1, 1) * M(2, 2);
        double c_jac1 =
                -2.0 * M(0, 2) * M_jac1(0, 2) - 2.0 * M(1, 2) * M_jac1(1, 2) - 2.0 * M(0, 1) * M_jac1(0, 1)
                + M_jac1(0, 0) * M(1, 1) + M(0, 0) * M_jac1(1, 1) + M_jac1(0, 0) * M(2, 2)
                + M(0, 0) * M_jac1(2, 2) + M_jac1(1, 1) * M(2, 2) + M(1, 1) * M_jac1(2, 2);
        double c_jac2 =
                -2.0 * M(0, 2) * M_jac2(0, 2) - 2.0 * M(1, 2) * M_jac2(1, 2) - 2.0 * M(0, 1) * M_jac2(0, 1)
                + M_jac2(0, 0) * M(1, 1) + M(0, 0) * M_jac2(1, 1) + M_jac2(0, 0) * M(2, 2)
                + M(0, 0) * M_jac2(2, 2) + M_jac2(1, 1) * M(2, 2) + M(1, 1) * M_jac2(2, 2);
        double c_jac3 =
                -2.0 * M(0, 2) * M_jac3(0, 2) - 2.0 * M(1, 2) * M_jac3(1, 2) - 2.0 * M(0, 1) * M_jac3(0, 1)
                + M_jac3(0, 0) * M(1, 1) + M(0, 0) * M_jac3(1, 1) + M_jac3(0, 0) * M(2, 2)
                + M(0, 0) * M_jac3(2, 2) + M_jac3(1, 1) * M(2, 2) + M(1, 1) * M_jac3(2, 2);
        double d =
                M(1, 1) * pow(M(0, 2), 2) + M(0, 0) * pow(M(1, 2), 2) + M(2, 2) * pow(M(0, 1), 2) -
                M(0, 0) * M(1, 1) * M(2, 2) - 2 * M(0, 1) * M(1, 2) * M(0, 2);
        double d_jac1 =
                M_jac1(1, 1) * pow(M(0, 2), 2) + M(1, 1) * 2 * M(0, 2) * M_jac1(0, 2)
                + M_jac1(0, 0) * pow(M(1, 2), 2) + M(0, 0) * 2.0 * M(1, 2) * M_jac1(1, 2)
                + M_jac1(2, 2) * pow(M(0, 1), 2) + M(2, 2) * 2.0 * M(0, 1) * M_jac1(0, 1)
                - M_jac1(0, 0) * M(1, 1) * M(2, 2) - M(0, 0) * M_jac1(1, 1) * M(2, 2)
                - M(0, 0) * M(1, 1) * M_jac1(2, 2) - 2.0 * (M_jac1(0, 1) * M(1, 2) * M(0, 2)
                                                            + M(0, 1) * M_jac1(1, 2) * M(0, 2) +
                                                            M(0, 1) * M(1, 2) * M_jac1(0, 2));
        double d_jac2 =
                M_jac2(1, 1) * pow(M(0, 2), 2) + M(1, 1) * 2 * M(0, 2) * M_jac2(0, 2)
                + M_jac2(0, 0) * pow(M(1, 2), 2) + M(0, 0) * 2.0 * M(1, 2) * M_jac2(1, 2)
                + M_jac2(2, 2) * pow(M(0, 1), 2) + M(2, 2) * 2.0 * M(0, 1) * M_jac2(0, 1)
                - M_jac2(0, 0) * M(1, 1) * M(2, 2) - M(0, 0) * M_jac2(1, 1) * M(2, 2)
                - M(0, 0) * M(1, 1) * M_jac2(2, 2) - 2.0 * (M_jac2(0, 1) * M(1, 2) * M(0, 2)
                                                            + M(0, 1) * M_jac2(1, 2) * M(0, 2) +
                                                            M(0, 1) * M(1, 2) * M_jac2(0, 2));
        double d_jac3 =
                M_jac3(1, 1) * pow(M(0, 2), 2) + M(1, 1) * 2 * M(0, 2) * M_jac3(0, 2)
                + M_jac3(0, 0) * pow(M(1, 2), 2) + M(0, 0) * 2.0 * M(1, 2) * M_jac3(1, 2)
                + M_jac3(2, 2) * pow(M(0, 1), 2) + M(2, 2) * 2.0 * M(0, 1) * M_jac3(0, 1)
                - M_jac3(0, 0) * M(1, 1) * M(2, 2) - M(0, 0) * M_jac3(1, 1) * M(2, 2)
                - M(0, 0) * M(1, 1) * M_jac3(2, 2) - 2.0 * (M_jac3(0, 1) * M(1, 2) * M(0, 2)
                                                            + M(0, 1) * M_jac3(1, 2) * M(0, 2) +
                                                            M(0, 1) * M(1, 2) * M_jac3(0, 2));

        double s = 2 * pow(b, 3) - 9 * b * c + 27 * d;
        double t = 4 * pow((pow(b, 2) - 3 * c), 3);

        double s_jac1 = 2.0 * 3.0 * pow(b, 2) * b_jac1 - 9.0 * b_jac1 * c - 9.0 * b * c_jac1 + 27.0 * d_jac1;
        double s_jac2 = 2.0 * 3.0 * pow(b, 2) * b_jac2 - 9.0 * b_jac2 * c - 9.0 * b * c_jac2 + 27.0 * d_jac2;
        double s_jac3 = 2.0 * 3.0 * pow(b, 2) * b_jac3 - 9.0 * b_jac3 * c - 9.0 * b * c_jac3 + 27.0 * d_jac3;
        double t_jac1 = 4.0 * 3.0 * pow((pow(b, 2) - 3.0 * c), 2) * (2.0 * b * b_jac1 - 3.0 * c_jac1);
        double t_jac2 = 4.0 * 3.0 * pow((pow(b, 2) - 3.0 * c), 2) * (2.0 * b * b_jac2 - 3.0 * c_jac2);
        double t_jac3 = 4.0 * 3.0 * pow((pow(b, 2) - 3.0 * c), 2) * (2.0 * b * b_jac3 - 3.0 * c_jac3);

        double alpha = acos(s / sqrt(t));
        double alpha_jac1 =
                -1.0 / sqrt(1.0 - (pow(s, 2) / t)) *
                (s_jac1 * sqrt(t) - s * 0.5 * pow(t, -0.5) * t_jac1) / t;
        double alpha_jac2 =
                -1.0 / sqrt(1.0 - (pow(s, 2) / t)) *
                (s_jac2 * sqrt(t) - s * 0.5 * pow(t, -0.5) * t_jac2) / t;
        double alpha_jac3 =
                -1.0 / sqrt(1.0 - (pow(s, 2) / t)) *
                (s_jac3 * sqrt(t) - s * 0.5 * pow(t, -0.5) * t_jac3) / t;
        double beta = alpha / 3;
        double beta_jac1 = alpha_jac1 / 3.0;
        double beta_jac2 = alpha_jac2 / 3.0;
        double beta_jac3 = alpha_jac3 / 3.0;
        double y = cos(beta);
        double y_jac1 = -sin(beta) * beta_jac1;
        double y_jac2 = -sin(beta) * beta_jac2;
        double y_jac3 = -sin(beta) * beta_jac3;

        double r = 0.5 * sqrt(t);
        double r_jac1 = 0.25 * pow(t, -0.5) * t_jac1;
        double r_jac2 = 0.25 * pow(t, -0.5) * t_jac2;
        double r_jac3 = 0.25 * pow(t, -0.5) * t_jac3;
        double w = pow(r, (1.0 / 3.0));
        double w_jac1 = (1.0 / 3.0) * pow(r, -2.0 / 3.0) * r_jac1;
        double w_jac2 = (1.0 / 3.0) * pow(r, -2.0 / 3.0) * r_jac2;
        double w_jac3 = (1.0 / 3.0) * pow(r, -2.0 / 3.0) * r_jac3;

        double k = w * y;
        double k_jac1 = w_jac1 * y + w * y_jac1;
        double k_jac2 = w_jac2 * y + w * y_jac2;
        double k_jac3 = w_jac3 * y + w * y_jac3;
        double smallestEV = (-b - 2 * k) / 3;
        double smallestEV_jac1 = (-b_jac1 - 2.0 * k_jac1) / 3.0;
        double smallestEV_jac2 = (-b_jac2 - 2.0 * k_jac2) / 3.0;
        double smallestEV_jac3 = (-b_jac3 - 2.0 * k_jac3) / 3.0;

        jacobian(0, 0) = smallestEV_jac1;
        jacobian(0, 1) = smallestEV_jac2;
        jacobian(0, 2) = smallestEV_jac3;
        return smallestEV;
    }

    template <typename T>
    static T GetSmallestEVwithJacobian(
            const Eigen::Matrix3d &xxF,
            const Eigen::Matrix3d &yyF,
            const Eigen::Matrix3d &zzF,
            const Eigen::Matrix3d &xyF,
            const Eigen::Matrix3d &yzF,
            const Eigen::Matrix3d &zxF,
            const Eigen::Matrix<T, 3, 1> &cayley,
            Eigen::Matrix<T, 1, 3> &jacobian) {


        using Matrix3T = Eigen::Matrix<T, 3, 3>;


        Matrix3T M_jac1 = Matrix3T::Zero();
        Matrix3T M_jac2 = Matrix3T::Zero();
        Matrix3T M_jac3 = Matrix3T::Zero();

        Matrix3T M = ComposeMwithJacobians(
                xxF, yyF, zzF, xyF, yzF, zxF, cayley, M_jac1, M_jac2, M_jac3);

        //Retrieve the smallest Eigenvalue by the following closed form solution.
        //Plus Jacobian.
        T b = -M(0, 0) - M(1, 1) - M(2, 2);
        T b_jac1 = -M_jac1(0, 0) - M_jac1(1, 1) - M_jac1(2, 2);
        T b_jac2 = -M_jac2(0, 0) - M_jac2(1, 1) - M_jac2(2, 2);
        T b_jac3 = -M_jac3(0, 0) - M_jac3(1, 1) - M_jac3(2, 2);
        T c =
                -ceres::pow(M(0, 2), 2) - ceres::pow(M(1, 2), 2) - ceres::pow(M(0, 1), 2) +
                M(0, 0) * M(1, 1) + M(0, 0) * M(2, 2) + M(1, 1) * M(2, 2);
        T c_jac1 =
                -2.0 * M(0, 2) * M_jac1(0, 2) - T(2.0) * M(1, 2) * M_jac1(1, 2) - 2.0 * M(0, 1) * M_jac1(0, 1)
                + M_jac1(0, 0) * M(1, 1) + M(0, 0) * M_jac1(1, 1) + M_jac1(0, 0) * M(2, 2)
                + M(0, 0) * M_jac1(2, 2) + M_jac1(1, 1) * M(2, 2) + M(1, 1) * M_jac1(2, 2);
        T c_jac2 =
                -2.0 * M(0, 2) * M_jac2(0, 2) - T(2.0) * M(1, 2) * M_jac2(1, 2) - 2.0 * M(0, 1) * M_jac2(0, 1)
                + M_jac2(0, 0) * M(1, 1) + M(0, 0) * M_jac2(1, 1) + M_jac2(0, 0) * M(2, 2)
                + M(0, 0) * M_jac2(2, 2) + M_jac2(1, 1) * M(2, 2) + M(1, 1) * M_jac2(2, 2);
        T c_jac3 =
                -2.0 * M(0, 2) * M_jac3(0, 2) - T(2.0) * M(1, 2) * M_jac3(1, 2) - 2.0 * M(0, 1) * M_jac3(0, 1)
                + M_jac3(0, 0) * M(1, 1) + M(0, 0) * M_jac3(1, 1) + M_jac3(0, 0) * M(2, 2)
                + M(0, 0) * M_jac3(2, 2) + M_jac3(1, 1) * M(2, 2) + M(1, 1) * M_jac3(2, 2);
        T d =
                M(1, 1) * ceres::pow(M(0, 2), 2) + M(0, 0) * ceres::pow(M(1, 2), 2) + M(2, 2) * ceres::pow(M(0, 1), 2) -
                M(0, 0) * M(1, 1) * M(2, 2) - T(2.0) * M(0, 1) * M(1, 2) * M(0, 2);
        T d_jac1 =
                M_jac1(1, 1) * ceres::pow(M(0, 2), 2) + M(1, 1) * T(2.0) * M(0, 2) * M_jac1(0, 2)
                + M_jac1(0, 0) * ceres::pow(M(1, 2), 2) + M(0, 0) * 2.0 * M(1, 2) * M_jac1(1, 2)
                + M_jac1(2, 2) * ceres::pow(M(0, 1), 2) + M(2, 2) * 2.0 * M(0, 1) * M_jac1(0, 1)
                - M_jac1(0, 0) * M(1, 1) * M(2, 2) - M(0, 0) * M_jac1(1, 1) * M(2, 2)
                - M(0, 0) * M(1, 1) * M_jac1(2, 2) - T(2.0) * (M_jac1(0, 1) * M(1, 2) * M(0, 2)
                                                            + M(0, 1) * M_jac1(1, 2) * M(0, 2) +
                                                            M(0, 1) * M(1, 2) * M_jac1(0, 2));
        T d_jac2 =
                M_jac2(1, 1) * ceres::pow(M(0, 2), 2) + M(1, 1) * T(2.0) * M(0, 2) * M_jac2(0, 2)
                + M_jac2(0, 0) * ceres::pow(M(1, 2), 2) + M(0, 0) * 2.0 * M(1, 2) * M_jac2(1, 2)
                + M_jac2(2, 2) * ceres::pow(M(0, 1), 2) + M(2, 2) * 2.0 * M(0, 1) * M_jac2(0, 1)
                - M_jac2(0, 0) * M(1, 1) * M(2, 2) - M(0, 0) * M_jac2(1, 1) * M(2, 2)
                - M(0, 0) * M(1, 1) * M_jac2(2, 2) - T(2.0) * (M_jac2(0, 1) * M(1, 2) * M(0, 2)
                                                            + M(0, 1) * M_jac2(1, 2) * M(0, 2) +
                                                            M(0, 1) * M(1, 2) * M_jac2(0, 2));
        T d_jac3 =
                M_jac3(1, 1) * ceres::pow(M(0, 2), 2) + M(1, 1) * T(2.0) * M(0, 2) * M_jac3(0, 2)
                + M_jac3(0, 0) * ceres::pow(M(1, 2), 2) + M(0, 0) * T(2.0) * M(1, 2) * M_jac3(1, 2)
                + M_jac3(2, 2) * ceres::pow(M(0, 1), 2) + M(2, 2) * 2.0 * M(0, 1) * M_jac3(0, 1)
                - M_jac3(0, 0) * M(1, 1) * M(2, 2) - M(0, 0) * M_jac3(1, 1) * M(2, 2)
                - M(0, 0) * M(1, 1) * M_jac3(2, 2) - T(2.0) * (M_jac3(0, 1) * M(1, 2) * M(0, 2)
                                                            + M(0, 1) * M_jac3(1, 2) * M(0, 2) +
                                                            M(0, 1) * M(1, 2) * M_jac3(0, 2));

        T s = T(2) * ceres::pow(b, 3) - T(9) * b * c + T(27) * d;
        T t = T(4) * ceres::pow((ceres::pow(b, 2) - T(3) * c), 3);

        T s_jac1 = T(2.0) * T(3.0) * ceres::pow(b, 2) * b_jac1 - T(9.0) * b_jac1 * c - T(9.0) * b * c_jac1 + T(27.0) * d_jac1;
        T s_jac2 = T(2.0) * T(3.0) * ceres::pow(b, 2) * b_jac2 - T(9.0) * b_jac2 * c - T(9.0) * b * c_jac2 + T(27.0) * d_jac2;
        T s_jac3 = T(2.0) * T(3.0) * ceres::pow(b, 2) * b_jac3 - T(9.0) * b_jac3 * c - T(9.0) * b * c_jac3 + T(27.0) * d_jac3;
        T t_jac1 = T(4.0) * T(3.0) * ceres::pow((ceres::pow(b, 2) - T(3.0) * c), 2) * (T(2.0) * b * b_jac1 - T(3.0) * c_jac1);
        T t_jac2 = T(4.0) * T(3.0) * ceres::pow((ceres::pow(b, 2) - T(3.0) * c), 2) * (T(2.0) * b * b_jac2 - T(3.0) * c_jac2);
        T t_jac3 = T(4.0) * T(3.0) * ceres::pow((ceres::pow(b, 2) - T(3.0) * c), 2) * (T(2.0) * b * b_jac3 - T(3.0) * c_jac3);

        T alpha = ceres::acos(s / ceres::sqrt(t));
        T alpha_jac1 =
                T(-1.0) / ceres::sqrt(T(1.0) - (ceres::pow(s, 2) / t)) *
                (s_jac1 * ceres::sqrt(t) - s * T(0.5) * ceres::pow(t, -0.5) * t_jac1) / t;
        T alpha_jac2 =
                T(-1.0) / ceres::sqrt(T(1.0) - (ceres::pow(s, 2) / t)) *
                (s_jac2 * ceres::sqrt(t) - s * T(0.5) * ceres::pow(t, -0.5) * t_jac2) / t;
        T alpha_jac3 =
                T(-1.0) / ceres::sqrt(T(1.0) - (ceres::pow(s, 2) / t)) *
                (s_jac3 * ceres::sqrt(t) - s * T(0.5) * ceres::pow(t, -0.5) * t_jac3) / t;
        T beta = alpha / T(3.0);
        T beta_jac1 = alpha_jac1 / T(3.0);
        T beta_jac2 = alpha_jac2 / T(3.0);
        T beta_jac3 = alpha_jac3 / T(3.0);
        T y = ceres::cos(beta);
        T y_jac1 = -ceres::sin(beta) * beta_jac1;
        T y_jac2 = -ceres::sin(beta) * beta_jac2;
        T y_jac3 = -ceres::sin(beta) * beta_jac3;

        T r = T(0.5) * ceres::sqrt(t);
        T r_jac1 = T(0.25) * ceres::pow(t, -0.5) * t_jac1;
        T r_jac2 = T(0.25) * ceres::pow(t, -0.5) * t_jac2;
        T r_jac3 = T(0.25) * ceres::pow(t, -0.5) * t_jac3;
        T w = ceres::pow(r, (1.0 / 3.0));
        T w_jac1 = (T(1.0) / T(3.0)) * ceres::pow(r, -2.0 / 3.0) * r_jac1;
        T w_jac2 = (T(1.0) / T(3.0)) * ceres::pow(r, -2.0 / 3.0) * r_jac2;
        T w_jac3 = (T(1.0) / T(3.0)) * ceres::pow(r, -2.0 / 3.0) * r_jac3;

        T k = w * y;
        T k_jac1 = w_jac1 * y + w * y_jac1;
        T k_jac2 = w_jac2 * y + w * y_jac2;
        T k_jac3 = w_jac3 * y + w * y_jac3;
        T smallestEV = (-b - T(2.0) * k) / T(3.0);

        T smallestEV_jac1 = (-b_jac1 - T(2.0) * k_jac1) / T(3.0);
        T smallestEV_jac2 = (-b_jac2 - T(2.0) * k_jac2) / T(3.0);
        T smallestEV_jac3 = (-b_jac3 - T(2.0) * k_jac3) / T(3.0);

        jacobian(0, 0) = smallestEV_jac1;
        jacobian(0, 1) = smallestEV_jac2;
        jacobian(0, 2) = smallestEV_jac3;
        return smallestEV;
    }

    static Eigen::Matrix3d ComposeM(
            const Eigen::Matrix3d &xxF,
            const Eigen::Matrix3d &yyF,
            const Eigen::Matrix3d &zzF,
            const Eigen::Matrix3d &xyF,
            const Eigen::Matrix3d &yzF,
            const Eigen::Matrix3d &zxF,
            const Eigen::Vector3d &cayley) {
        Eigen::Matrix3d M;
        Eigen::Matrix3d R = Cayley2RotReduced(cayley);

        //Fill the matrix M using the precomputed summation terms
        double temp;
        temp = R.row(2) * yyF * R.row(2).transpose();
        M(0, 0) = temp;
        temp = -2.0 * R.row(2) * yzF * R.row(1).transpose();
        M(0, 0) += temp;
        temp = R.row(1) * zzF * R.row(1).transpose();
        M(0, 0) += temp;

        temp = R.row(2) * yzF * R.row(0).transpose();
        M(0, 1) = temp;
        temp = -1.0 * R.row(2) * xyF * R.row(2).transpose();
        M(0, 1) += temp;
        temp = -1.0 * R.row(1) * zzF * R.row(0).transpose();
        M(0, 1) += temp;
        temp = R.row(1) * zxF * R.row(2).transpose();
        M(0, 1) += temp;

        temp = R.row(2) * xyF * R.row(1).transpose();
        M(0, 2) = temp;
        temp = -1.0 * R.row(2) * yyF * R.row(0).transpose();
        M(0, 2) += temp;
        temp = -1.0 * R.row(1) * zxF * R.row(1).transpose();
        M(0, 2) += temp;
        temp = R.row(1) * yzF * R.row(0).transpose();
        M(0, 2) += temp;

        temp = R.row(0) * zzF * R.row(0).transpose();
        M(1, 1) = temp;
        temp = -2.0 * R.row(0) * zxF * R.row(2).transpose();
        M(1, 1) += temp;
        temp = R.row(2) * xxF * R.row(2).transpose();
        M(1, 1) += temp;

        temp = R.row(0) * zxF * R.row(1).transpose();
        M(1, 2) = temp;
        temp = -1.0 * R.row(0) * yzF * R.row(0).transpose();
        M(1, 2) += temp;
        temp = -1.0 * R.row(2) * xxF * R.row(1).transpose();
        M(1, 2) += temp;
        temp = R.row(2) * xyF * R.row(0).transpose();
        M(1, 2) += temp;

        temp = R.row(1) * xxF * R.row(1).transpose();
        M(2, 2) = temp;
        temp = -2.0 * R.row(0) * xyF * R.row(1).transpose();
        M(2, 2) += temp;
        temp = R.row(0) * yyF * R.row(0).transpose();
        M(2, 2) += temp;

        M(1, 0) = M(0, 1);
        M(2, 0) = M(0, 2);
        M(2, 1) = M(1, 2);

        return M;
    }
}

#endif //OPENGV_EIGEN_SOLVER_HPP
