/*
 * @Author: Yijia He 
 * @Date: 2021-11-06 17:26:45 
 * @Last Modified by: Yijia He
 * @Last Modified time: 2021-11-07 18:38:54
 */

#include "IMU/imuPreintegrated.hpp"
#include "utils/sophusExtUtils.hpp"

namespace vio {

    IMUPreintegrated::IMUPreintegrated(const IMUBias &bias, const IMUCalibParam *calib, double start_time,
                                       double end_time) {
        Nga_ = calib->cov_;
        NgaWalk_ = calib->cov_walk_;
        initialize(bias);
        start_t_ns = start_time;
        end_t_ns = end_time;
    }

    void IMUPreintegrated::initialize(const IMUBias &bias) {
        sum_dt_ = 0.0f;
        imu_measurements_.clear();
        dR_ = Sophus::SO3d();
        dV_.setZero();
        dP_.setZero();
        JRg_.setZero();
        JVg_.setZero();
        JVa_.setZero();
        JPg_.setZero();
        JPa_.setZero();
        bias_ = bias;
        Cov_.setZero();
    }

    void IMUPreintegrated::reinitialize(const IMUBias &bias)
    {
        sum_dt_ = 0.0f;
        dR_ = Sophus::SO3d();
        dV_.setZero();
        dP_.setZero();
        JRg_.setZero();
        JVg_.setZero();
        JVa_.setZero();
        JPg_.setZero();
        JPa_.setZero();
        bias_ = bias;
        Cov_.setZero();
    }

/// @cite Christian Forster, Luca Carlone, Frank Dellaert, Davide Scaramuzza, “On-Manifold Preintegration for Real-Time Visual-Inertial Odometry”, in IEEE Transactions on Robotics, 2016.
    void IMUPreintegrated::integrate_new_measurement(const Eigen::Vector3d &gyro_mea, const Eigen::Vector3d &acc_mea,
                                                     const double &dt) {
        // Position is updated firstly, as it depends on previously computed velocity and rotation.
        // Velocity is updated secondly, as it depends on previously computed rotation.
        // Rotation is the last to be updated.

        imu_measurements_.push_back(integrable(gyro_mea, acc_mea, dt));
        // For the matrix A, B please see equation (62)
        Eigen::Matrix<double, 9, 9> A = Eigen::Matrix<double, 9, 9>::Identity();  // 0-2:phi, 3-5: vel, 6-8: pos
        Eigen::Matrix<double, 9, 6> B = Eigen::Matrix<double, 9, 6>::Zero();      // 0-2: \eta^g, 3-5: \eta^a

        Eigen::Vector3d acc = acc_mea - bias_.ba_;
        Eigen::Vector3d gyro = gyro_mea - bias_.bg_;
        double dt2 = dt * dt;

        /// step 1: Update delta position dP and velocity dV (rely on no-updated delta rotation)
        Eigen::Matrix3d mdR = dR_.matrix();
        dP_ = dP_ + dV_ * dt + 0.5f * mdR * acc * dt2;
        dV_ = dV_ + mdR * acc * dt;

        // 1.1 Compute velocity and position parts of matrices A and B (rely on non-updated delta rotation)
        Eigen::Matrix3d acc_skew;
        acc_skew << 0.0, -acc.z(), acc.y(),
                acc.z(), 0.0, -acc.x(),
                -acc.y(), acc.x(), 0.0;
        A.block<3, 3>(3, 0) = -mdR * acc_skew * dt;              // eq.(60)  dv/dphi
        A.block<3, 3>(6, 0) = 0.5f * dt * A.block<3, 3>(3, 0);     // eq.(61)  dp/dphi
        A.block<3, 3>(6, 3) = Eigen::Matrix3d::Identity() * dt;  // eq.(61)  dp/dv
        B.block<3, 3>(3, 3) = mdR * dt;                          // eq.(60)  dv/deta
        B.block<3, 3>(6, 3) = 0.5f * dt2 * mdR;                 // eq.(61)  dp/deta

        // 1.2 Update position and velocity jacobians wrt bias correction
        JPa_ = JPa_ + JVa_ * dt - B.block<3, 3>(6, 3);
        JPg_ = JPg_ + JVg_ * dt + A.block<3, 3>(6, 0) * JRg_;
        JVa_ = JVa_ - B.block<3, 3>(3, 3);                       // eq.(69)  dv/dba
        JVg_ = JVg_ + A.block<3, 3>(3, 0) * JRg_;                // eq.(69)  dv/dbg

        /// step 2: Update delta rotation
        Eigen::Vector3d phi = gyro * dt;
        Sophus::SO3d dRi = Sophus::SO3d::exp(phi);
        dR_ = dR_ * dRi;
        dR_.normalize();

        // 2.1 Compute rotation parts of matrices A and B
        A.block<3, 3>(0, 0) = dRi.matrix().transpose();         // eq.(59)   dphi/dphi
        Eigen::Matrix3d rightJ;
        Sophus::rightJacobianSO3(phi, rightJ);
        B.block<3, 3>(0, 0) = rightJ * dt;                      // eq.(59)   dphi/deta

        // 2.2 Update rotation jacobian wrt bias correction
        // ref: https://github.com/UZ-SLAMLab/ORB_SLAM3/issues/212#issuecomment-745138770
        JRg_ = A.block<3, 3>(0, 0) * JRg_ - B.block<3, 3>(0, 0);

        // step 3: Update covariance                          // eq.(63)
        Cov_.block<9, 9>(0, 0) =
                A * Cov_.block<9, 9>(0, 0) * A.transpose() + B * Nga_ * B.transpose();  // phi, vel, pos covairance
        Cov_.block<6, 6>(9, 9) = Cov_.block<6, 6>(9, 9) + NgaWalk_;                              // bias covariance

        //hyj, Force the matrix Cov_ to maintain symmetry
        Eigen::Matrix<double, 9, 9> cov = (Cov_.block<9, 9>(0, 0) + Cov_.block<9, 9>(0, 0).transpose()) / 2.0f;
        Cov_.block<9, 9>(0, 0) = cov;

        // Total integrated time
        sum_dt_ += dt;
    }

    Sophus::SO3d IMUPreintegrated::reintegrate_gyro_measurement(const Eigen::Vector3d& biasg) {

        Sophus::SO3d dR_only_ = Sophus::SO3d();

        for(int i = 0; i < imu_measurements_.size();i++)
        {
            Eigen::Vector3d gyro_mea = imu_measurements_[i].w_;
            double dt = imu_measurements_[i].t_;
            Eigen::Vector3d gyro = gyro_mea - biasg;
            Eigen::Vector3d phi = gyro * dt;
            Sophus::SO3d dRi = Sophus::SO3d::exp(phi);
            dR_only_ = dR_only_ * dRi;
            dR_only_.normalize();
        }

        return dR_only_;
    }

    void IMUPreintegrated::reintegrate(const IMUBias &bias) {

        // bx, set the original of integrate to zero
        reinitialize(bias);

        for(int i = 0; i < imu_measurements_.size(); i++) {

            Eigen::Vector3d gyro_mea = imu_measurements_[i].w_;
            Eigen::Vector3d acc_mea = imu_measurements_[i].a_;
            double dt = imu_measurements_[i].t_;

            // For the matrix A, B please see equation (62)
            Eigen::Matrix<double, 9, 9> A = Eigen::Matrix<double, 9, 9>::Identity();  // 0-2:phi, 3-5: vel, 6-8: pos
            Eigen::Matrix<double, 9, 6> B = Eigen::Matrix<double, 9, 6>::Zero();      // 0-2: \eta^g, 3-5: \eta^a

            Eigen::Vector3d acc = acc_mea - bias_.ba_;
            Eigen::Vector3d gyro = gyro_mea - bias_.bg_;
            double dt2 = dt * dt;

            /// step 1: Update delta position dP and velocity dV (rely on no-updated delta rotation)
            Eigen::Matrix3d mdR = dR_.matrix();
            dP_ = dP_ + dV_ * dt + 0.5f * mdR * acc * dt2;
            dV_ = dV_ + mdR * acc * dt;

            // 1.1 Compute velocity and position parts of matrices A and B (rely on non-updated delta rotation)
            Eigen::Matrix3d acc_skew;
            acc_skew << 0.0, -acc.z(), acc.y(),
                    acc.z(), 0.0, -acc.x(),
                    -acc.y(), acc.x(), 0.0;
            A.block<3, 3>(3, 0) = -mdR * acc_skew * dt;              // eq.(60)  dv/dphi
            A.block<3, 3>(6, 0) = 0.5f * dt * A.block<3, 3>(3, 0);     // eq.(61)  dp/dphi
            A.block<3, 3>(6, 3) = Eigen::Matrix3d::Identity() * dt;  // eq.(61)  dp/dv
            B.block<3, 3>(3, 3) = mdR * dt;                          // eq.(60)  dv/deta
            B.block<3, 3>(6, 3) = 0.5f * dt2 * mdR;                 // eq.(61)  dp/deta

            // 1.2 Update position and velocity jacobians wrt bias correction
            JPa_ = JPa_ + JVa_ * dt - B.block<3, 3>(6, 3);
            JPg_ = JPg_ + JVg_ * dt + A.block<3, 3>(6, 0) * JRg_;
            JVa_ = JVa_ - B.block<3, 3>(3, 3);                       // eq.(69)  dv/dba
            JVg_ = JVg_ + A.block<3, 3>(3, 0) * JRg_;                // eq.(69)  dv/dbg

            /// step 2: Update delta rotation
            Eigen::Vector3d phi = gyro * dt;
            Sophus::SO3d dRi = Sophus::SO3d::exp(phi);
            dR_ = dR_ * dRi;
            dR_.normalize();

            // 2.1 Compute rotation parts of matrices A and B
            A.block<3, 3>(0, 0) = dRi.matrix().transpose();         // eq.(59)   dphi/dphi
            Eigen::Matrix3d rightJ;
            Sophus::rightJacobianSO3(phi, rightJ);
            B.block<3, 3>(0, 0) = rightJ * dt;                      // eq.(59)   dphi/deta

            // 2.2 Update rotation jacobian wrt bias correction
            // ref: https://github.com/UZ-SLAMLab/ORB_SLAM3/issues/212#issuecomment-745138770
            JRg_ = A.block<3, 3>(0, 0) * JRg_ - B.block<3, 3>(0, 0);

            // step 3: Update covariance                          // eq.(63)
            Cov_.block<9, 9>(0, 0) =
                    A * Cov_.block<9, 9>(0, 0) * A.transpose() + B * Nga_ * B.transpose();  // phi, vel, pos covairance
            Cov_.block<6, 6>(9, 9) = Cov_.block<6, 6>(9, 9) + NgaWalk_;                              // bias covariance

            //hyj, Force the matrix Cov_ to maintain symmetry
            Eigen::Matrix<double, 9, 9> cov = (Cov_.block<9, 9>(0, 0) + Cov_.block<9, 9>(0, 0).transpose()) / 2.0f;
            Cov_.block<9, 9>(0, 0) = cov;

            // Total integrated time
            sum_dt_ += dt;
        }
    }

    void IMUPreintegrated::reintegrate(const IMUBias &bias, const std::vector<integrable>& new_imu_meas) {

        // bx, set the original of integrate to zero
        reinitialize(bias);

        for(int i = 0; i < new_imu_meas.size(); i++) {

            Eigen::Vector3d gyro_mea = new_imu_meas[i].w_;
            Eigen::Vector3d acc_mea = new_imu_meas[i].a_;
            double dt = new_imu_meas[i].t_;

            // For the matrix A, B please see equation (62)
            Eigen::Matrix<double, 9, 9> A = Eigen::Matrix<double, 9, 9>::Identity();  // 0-2:phi, 3-5: vel, 6-8: pos
            Eigen::Matrix<double, 9, 6> B = Eigen::Matrix<double, 9, 6>::Zero();      // 0-2: \eta^g, 3-5: \eta^a

            Eigen::Vector3d acc = acc_mea - bias_.ba_;
            Eigen::Vector3d gyro = gyro_mea - bias_.bg_;
            double dt2 = dt * dt;

            /// step 1: Update delta position dP and velocity dV (rely on no-updated delta rotation)
            Eigen::Matrix3d mdR = dR_.matrix();
            dP_ = dP_ + dV_ * dt + 0.5f * mdR * acc * dt2;
            dV_ = dV_ + mdR * acc * dt;

            // 1.1 Compute velocity and position parts of matrices A and B (rely on non-updated delta rotation)
            Eigen::Matrix3d acc_skew;
            acc_skew << 0.0, -acc.z(), acc.y(),
                    acc.z(), 0.0, -acc.x(),
                    -acc.y(), acc.x(), 0.0;
            A.block<3, 3>(3, 0) = -mdR * acc_skew * dt;              // eq.(60)  dv/dphi
            A.block<3, 3>(6, 0) = 0.5f * dt * A.block<3, 3>(3, 0);     // eq.(61)  dp/dphi
            A.block<3, 3>(6, 3) = Eigen::Matrix3d::Identity() * dt;  // eq.(61)  dp/dv
            B.block<3, 3>(3, 3) = mdR * dt;                          // eq.(60)  dv/deta
            B.block<3, 3>(6, 3) = 0.5f * dt2 * mdR;                 // eq.(61)  dp/deta

            // 1.2 Update position and velocity jacobians wrt bias correction
            JPa_ = JPa_ + JVa_ * dt - B.block<3, 3>(6, 3);
            JPg_ = JPg_ + JVg_ * dt + A.block<3, 3>(6, 0) * JRg_;
            JVa_ = JVa_ - B.block<3, 3>(3, 3);                       // eq.(69)  dv/dba
            JVg_ = JVg_ + A.block<3, 3>(3, 0) * JRg_;                // eq.(69)  dv/dbg

            /// step 2: Update delta rotation
            Eigen::Vector3d phi = gyro * dt;
            Sophus::SO3d dRi = Sophus::SO3d::exp(phi);
            dR_ = dR_ * dRi;
            dR_.normalize();

            // 2.1 Compute rotation parts of matrices A and B
            A.block<3, 3>(0, 0) = dRi.matrix().transpose();         // eq.(59)   dphi/dphi
            Eigen::Matrix3d rightJ;
            Sophus::rightJacobianSO3(phi, rightJ);
            B.block<3, 3>(0, 0) = rightJ * dt;                      // eq.(59)   dphi/deta

            // 2.2 Update rotation jacobian wrt bias correction
            // ref: https://github.com/UZ-SLAMLab/ORB_SLAM3/issues/212#issuecomment-745138770
            JRg_ = A.block<3, 3>(0, 0) * JRg_ - B.block<3, 3>(0, 0);

            // step 3: Update covariance                          // eq.(63)
            Cov_.block<9, 9>(0, 0) =
                    A * Cov_.block<9, 9>(0, 0) * A.transpose() + B * Nga_ * B.transpose();  // phi, vel, pos covairance
            Cov_.block<6, 6>(9, 9) = Cov_.block<6, 6>(9, 9) + NgaWalk_;                              // bias covariance

            //hyj, Force the matrix Cov_ to maintain symmetry
            Eigen::Matrix<double, 9, 9> cov = (Cov_.block<9, 9>(0, 0) + Cov_.block<9, 9>(0, 0).transpose()) / 2.0f;
            Cov_.block<9, 9>(0, 0) = cov;

            // Total integrated time
            sum_dt_ += dt;
        }
    }

    Matrix15d IMUPreintegrated::get_information()  // TODO:: avoid re-computation.
    {
        // Info_ = Cov_.inverse();
        Info_.setZero();
        Info_.block<9, 9>(0, 0) = Cov_.block<9, 9>(0, 0).inverse();
        for (size_t i = 9; i < 15; i++) {
            Info_(i, i) = 1.0f / Cov_(i, i);
        }
        // std::cout << "compare informatrix:\n" << Info_.block<9,9>(0,0) << std::endl << Cov_.block<9,9>(0,0).inverse() << std::endl;
        return Info_;
    }

    Eigen::Vector3d IMUPreintegrated::get_delta_position(const IMUBias &new_bias) const{
        Eigen::Vector3d dbg = new_bias.bg_ - bias_.bg_;
        Eigen::Vector3d dba = new_bias.ba_ - bias_.ba_;
        return dP_ + JPg_ * dbg + JPa_ * dba;                    // eq.(44)
    }

    Eigen::Vector3d IMUPreintegrated::get_delta_velocity(const IMUBias &new_bias) const {
        Eigen::Vector3d dbg = new_bias.bg_ - bias_.bg_;
        Eigen::Vector3d dba = new_bias.ba_ - bias_.ba_;
        return dV_ + JVg_ * dbg + JVa_ * dba;                    // eq.(44)
    }

    Sophus::SO3d IMUPreintegrated::get_delta_rotation(const IMUBias &new_bias) const {
        Eigen::Vector3d dbg = new_bias.bg_ - bias_.bg_;
        Sophus::SO3d dR_update = dR_ * Sophus::SO3d::exp(JRg_ * dbg);
        // dR_update.normalize();                               // eq.(44)
        return dR_update;
    }

} // namespace vio