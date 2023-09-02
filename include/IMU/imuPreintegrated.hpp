/*
 * @Author: Yijia He
 * @Date: 2021-11-06 16:27:21 
 * @Last Modified by: Yijia He
 * @Last Modified time: 2021-11-07 18:37:37
 */

#ifndef IMUPREINTEGRATED_HPP
#define IMUPREINTEGRATED_HPP
#include "basicTypes.hpp"

namespace vio {

class IMUPreintegrated
{
public:

    struct integrable
    {

        integrable(const Eigen::Vector3d &w, const Eigen::Vector3d &a , const double &t):w_(w),a_(a),t_(t){}
        Eigen::Vector3d w_;
        Eigen::Vector3d a_;
        double t_;
    };

    IMUPreintegrated(){}
    IMUPreintegrated(const IMUBias &bias, const IMUCalibParam *calib, double start_time, double end_time);

    void initialize(const IMUBias &bias);
    void reinitialize(const IMUBias &bias);

    void integrate_new_measurement(const Eigen::Vector3d &gyro_mea, const Eigen::Vector3d &acc_mea, const double &dt);
    // TODO only reintegrate the roation with biasg
    Sophus::SO3d reintegrate_gyro_measurement(const Eigen::Vector3d& biasg);

    // TODO:: reintegrate imu data when delta bias large than threshold 
    void reintegrate(const IMUBias &bias);
    void reintegrate(const IMUBias &bias, const std::vector<integrable>& new_imu_meas);

    Matrix15d      get_information();
    Eigen::Vector3d get_delta_velocity(const IMUBias &new_bias) const;
    Eigen::Vector3d get_delta_position(const IMUBias &new_bias) const;
    Sophus::SO3d    get_delta_rotation(const IMUBias &new_bias) const;

    IMUBias bias_;

    double sum_dt_;        // time integration 
    Matrix6d Nga_;        // imu gauss noise covariance
    Matrix6d NgaWalk_;    // imu random walk covariance 
    Matrix15d Cov_;       // imu intergration covariance
    Matrix15d Info_;      // imu intergration information matrix
    
    // pre-integrate imu data
    Sophus::SO3d dR_;
    Eigen::Vector3d dV_;
    Eigen::Vector3d dP_;
    // jacobian 
    Eigen::Matrix3d JRg_;
    Eigen::Matrix3d JVg_;
    Eigen::Matrix3d JVa_;
    Eigen::Matrix3d JPg_;
    Eigen::Matrix3d JPa_;


    std::vector<integrable> imu_measurements_;

    // 使得IMU强行对齐
    double start_t_ns;
    double end_t_ns;


};

} // namespace vio

#endif