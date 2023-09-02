#ifndef IMU_INTEG_FACTOR_H
#define IMU_INTEG_FACTOR_H
#include <ceres/ceres.h>
#include "IMU/basicTypes.hpp"
#include "IMU/imuPreintegrated.hpp"

namespace vio {

class ImuIntegFactor : public ceres::SizedCostFunction<15, 6, 9, 6, 9> {
  public:
    ImuIntegFactor(IMUPreintegrated* pre_integration) : pre_integration_(pre_integration) {
        Eigen::Matrix<double, 15, 15> info = pre_integration_->get_information();
        sqrt_info_ = Eigen::LLT<Eigen::Matrix<double, 15, 15>>(info.cast<double>()).matrixL().transpose();
    }

    void ScaleSqrtInfo(double scale) {
        // only scale the sub information matrix for rotation, velocity, pose.
        sqrt_info_.block<9, 9>(0, 0) *= scale;
        sqrt_info_.block<6, 6>(9, 9) *= 1e-1;    // scale bias information.
    }

    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const;
    void check(double** parameters);

    Eigen::Matrix<double, 15, 15> sqrt_info_;
    IMUPreintegrated* pre_integration_;
};

class ImuIntegGdirFactor : public ceres::SizedCostFunction<15, 6, 9, 6, 9, 3> {
  public:
    ImuIntegGdirFactor(IMUPreintegrated* pre_integration) : pre_integration_(pre_integration) {
        Eigen::Matrix<double, 15, 15> info = pre_integration_->get_information();
        sqrt_info_ = Eigen::LLT<Eigen::Matrix<double, 15, 15>>(info.cast<double>()).matrixL().transpose();
    }

    void ScaleSqrtInfo(double scale) {
        // only scale the sub information matrix for rotation, velocity, pose.
        sqrt_info_.block<9, 9>(0, 0) *= scale;
        sqrt_info_.block<6, 6>(9, 9) *= 1e-1;    // scale bias information.
    }

    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const;
    void check(double** parameters);

    Eigen::Matrix<double, 15, 15> sqrt_info_;
    IMUPreintegrated* pre_integration_;
};

}
#endif /* IMU_INTEG_FACTOR_H */